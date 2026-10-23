"""Fixed train/validation/test data; all fitted transforms use training only."""
import argparse
import hashlib
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'checklist_v1'


def save(name, splits, metadata, labels=None):
    DATA.mkdir(parents=True, exist_ok=True)
    arr = {k: np.asarray(v, dtype=np.float32) for k, v in splits.items()}
    if labels:
        arr.update({k + '_labels': np.asarray(v, dtype=np.int64) for k, v in labels.items()})
    path = DATA / (name + '.npz')
    np.savez_compressed(path, **arr)
    metadata.update(shapes={k: list(v.shape) for k, v in arr.items()}, sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest())
    path.with_suffix('.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(name, metadata['shapes'], flush=True)


def toy(name, n, rng):
    if name == 'gmm8':
        a = rng.integers(8, size=n) * np.pi / 4
        return 2 * np.stack([np.cos(a), np.sin(a)], 1) + rng.normal(0, .12, (n, 2))
    if name == 'moons':
        a = rng.uniform(0, np.pi, n)
        k = rng.integers(2, size=n)
        return np.stack([np.where(k == 0, np.cos(a), 1 - np.cos(a)), np.where(k == 0, np.sin(a), .5 - np.sin(a))], 1) + rng.normal(0, .05, (n, 2))
    if name == 'scurve':
        a = rng.uniform(-1.5 * np.pi, 1.5 * np.pi, n)
        return np.stack([np.sin(a), np.sign(a) * (np.cos(a) - 1)], 1) + rng.normal(0, .03, (n, 2))
    if name == 'checkerboard':
        cell = rng.integers(0, 4, size=(n, 2))
        cell[:, 1] = 2 * (cell[:, 1] // 2) + cell[:, 0] % 2
        return (cell + rng.uniform(size=(n, 2))) - 2
    raise ValueError(name)


def synthetic():
    for i, name in enumerate(['gmm8', 'moons', 'scurve', 'checkerboard']):
        if (DATA / (name + '.npz')).exists():
            continue
        splits = {s: toy(name, n, np.random.default_rng(41000 + 100 * i + j)) for j, (s, n) in enumerate([('train', 100000), ('val', 10000), ('test', 10000)])}
        mu, std = splits['train'].mean(0), splits['train'].std(0)
        save(name, {s: (x - mu) / std for s, x in splits.items()}, dict(version=1, generator='benchmark/data.py:toy', split_seeds=[41000 + 100 * i + j for j in range(3)], mean=mu.tolist(), std=std.tolist()))
    for name in ['sine', 'lorenz']:
        if (DATA / (name + '.npz')).exists():
            continue
        splits = {}
        # Each window comes from a separately sampled trajectory. No overlapping trajectories across splits.
        for j, (s, n) in enumerate([('train', 10000), ('val', 2000), ('test', 2000)]):
            rng = np.random.default_rng(42000 + j)
            if name == 'sine':
                tau = np.arange(128) * .02
                freq = rng.uniform(.3, 2.5, (n, 1, 1, 2))
                phase = rng.uniform(-np.pi, np.pi, (n, 1, 1, 2))
                lag = np.array([0, .5, 1.])[:, None]
                amp = rng.uniform(.4, 1.4, (n, 1, 3, 2))
                x = (amp * np.sin(2 * np.pi * tau[None, :, None, None] * freq + phase + lag)).sum(-1)
                x += rng.normal(0, .025, x.shape)
            else:
                # Vectorized RK4 of independent Lorenz systems; discard 500 burn-in steps.
                state = rng.uniform(-15, 15, (n, 3))
                def f(z):
                    xx, yy, zz = z.T
                    return np.stack([10 * (yy - xx), xx * (28 - zz) - yy, xx * yy - (8 / 3) * zz], 1)
                trajectory = []
                for k in range(628):
                    a = f(state); b = f(state + .005 * a); c = f(state + .005 * b); d = f(state + .01 * c)
                    state += .01 / 6 * (a + 2 * b + 2 * c + d)
                    if k >= 500:
                        trajectory.append(state.copy())
                x = np.stack(trajectory, 1)
            splits[s] = x.astype('float32')
        mu, std = splits['train'].mean((0, 1)), splits['train'].std((0, 1))
        save(name, {s: (x - mu) / std for s, x in splits.items()}, dict(version=1, independent_trajectories=True, split_seeds=[42000, 42001, 42002], observation_dt=.02 if name == 'sine' else .01, generation_time=[0, 1], lorenz_parameters=[10, 28, 8/3] if name == 'lorenz' else None, mean=mu.tolist(), std=std.tolist()))


def images():
    from torchvision.datasets import MNIST, CIFAR10
    for name, cls in [('mnist', MNIST), ('cifar10', CIFAR10)]:
        if (DATA / (name + '.npz')).exists():
            continue
        tr = cls(str(DATA / 'downloads'), train=True, download=True)
        te = cls(str(DATA / 'downloads'), train=False, download=True)
        x, xt = np.asarray(tr.data), np.asarray(te.data)
        x = x[:, None] if name == 'mnist' else x.transpose(0, 3, 1, 2)
        xt = xt[:, None] if name == 'mnist' else xt.transpose(0, 3, 1, 2)
        idx = np.random.default_rng(43000).permutation(len(x))
        it, iv = idx[:-5000], idx[-5000:]
        labels = np.asarray(tr.targets)
        save(name, dict(train=x[it] / 127.5 - 1, val=x[iv] / 127.5 - 1, test=xt / 127.5 - 1), dict(version=1, source=cls.__module__, split_seed=43000, scale='uint8 / 127.5 - 1', train_indices=it.tolist(), val_indices=iv.tolist()), dict(train=labels[it], val=labels[iv], test=np.asarray(te.targets)))


def har():
    if (DATA / 'har.npz').exists():
        return
    url = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
    archive = DATA / 'downloads' / 'har.zip'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        urllib.request.urlretrieve(url, archive)
    with zipfile.ZipFile(archive) as z:
        inner = [n for n in z.namelist() if n.endswith('.zip')]
        if inner:
            import io
            z2 = zipfile.ZipFile(io.BytesIO(z.read(inner[0])))
        else:
            z2 = z
        prefix = next(n.split('train/')[0] for n in z2.namelist() if n.endswith('train/y_train.txt'))
        def read(split, f):
            return np.loadtxt(z2.open(prefix + split + '/' + f))
        def signals(split):
            return np.stack([read(split, 'Inertial Signals/' + k + '_' + a + '_' + split + '.txt') for k in ['total_acc', 'body_acc', 'body_gyro'] for a in 'xyz'], -1)
        x, xt = signals('train'), signals('test')
        subjects = read('train', 'subject_train.txt').astype(int)
        subjects_test = read('test', 'subject_test.txt').astype(int)
        valsubjects = [5, 16, 25, 26]
        val = np.isin(subjects, valsubjects)
        assert val.any() and not np.intersect1d(subjects, subjects_test).size
        y, yt = read('train', 'y_train.txt').astype(int)-1, read('test', 'y_test.txt').astype(int)-1
        mu, std = x[~val].mean((0, 1)), x[~val].std((0, 1))
        save('har', dict(train=(x[~val]-mu)/std, val=(x[val]-mu)/std, test=(xt-mu)/std), dict(version=1, url=url, train_subjects=np.unique(subjects[~val]).tolist(), val_subjects=valsubjects, test_subjects=np.unique(subjects_test).tolist(), channels=[k+'_'+a for k in ['total_acc', 'body_acc', 'body_gyro'] for a in 'xyz'], mean=mu.tolist(), std=std.tolist()), dict(train=y[~val], val=y[val], test=yt))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--kind', choices=['synthetic', 'images', 'har', 'all'], default='all'); a=p.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    for name, fn in [('synthetic', synthetic), ('images', images), ('har', har)]:
        if a.kind in [name, 'all']:
            fn()
