"""Frozen image evaluators for E03; trained exclusively on real training images."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from threadpoolctl import threadpool_limits
from .data import DATA,ROOT
from .protocol import RESULTS
from .run import write_json
from .metrics import frechet


class Classifier(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(channels,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(64,64,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),nn.Linear(1024,128),nn.ReLU())
        self.head=nn.Linear(128,10)
    def forward(self,x):
        feat=self.body(x)
        return self.head(feat),feat


@torch.no_grad()
def extract(model,x):
    model.eval(); feats=[]; logits=[]
    for lo in range(0,len(x),256):
        a,b=model(torch.as_tensor(x[lo:lo+256],device='cuda').clamp(-1,1));logits.append(a.cpu());feats.append(b.cpu())
    return torch.cat(logits).numpy(),torch.cat(feats).numpy()


def train_classifier(dataset):
    folder=RESULTS/'evaluators'/dataset;folder.mkdir(parents=True,exist_ok=True)
    path=folder/'classifier.pt'
    if path.exists():return
    torch.manual_seed(91000);torch.cuda.manual_seed_all(91000)
    with np.load(DATA/(dataset+'.npz')) as z:arrays={k:z[k] for k in z.files}
    model=Classifier(arrays['train'].shape[1]).cuda();opt=torch.optim.Adam(model.parameters(),lr=.001)
    x=torch.as_tensor(arrays['train'],device='cuda');y=torch.as_tensor(arrays['train_labels'],device='cuda')
    gen=torch.Generator(device='cuda').manual_seed(91000);best=-1;records=[];start=time.perf_counter()
    for epoch in range(5 if dataset=='mnist' else 20):
        model.train();idx=torch.randperm(len(x),generator=gen,device='cuda')
        for lo in range(0,len(x),128):
            ids=idx[lo:lo+128];opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[ids])[0],y[ids]);loss.backward();opt.step()
        logits,_=extract(model,arrays['val']);acc=float(np.mean(logits.argmax(-1)==arrays['val_labels']));records.append(dict(epoch=epoch+1,val_accuracy=acc))
        if acc>best:
            best=acc;torch.save(dict(model=model.state_dict(),epoch=epoch+1),folder/'candidate.pt')
        print('classifier',dataset,epoch+1,acc,flush=True)
    ck=torch.load(folder/'candidate.pt',weights_only=True);model.load_state_dict(ck['model']);logits,feats=extract(model,arrays['test'])
    torch.save(ck,path);np.savez_compressed(folder/'reference_features.npz',features=feats,logits=logits)
    write_json(folder/'config.json',dict(seed=91000,dataset=dataset,training_only_real=True,checkpoint_rule='maximum real validation accuracy',test_accuracy=float(np.mean(logits.argmax(-1)==arrays['test_labels'])),features=128,feature_layer='last ReLU before 10-class head',records=records,wall_seconds=time.perf_counter()-start,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),metric_name='FD-MNIST' if dataset=='mnist' else 'CIFAR class-distribution diagnostic only; FID uses Inception'))


class Uint8Images(Dataset):
    def __init__(self,x):self.x=x
    def __len__(self):return len(self.x)
    def __getitem__(self,i):return torch.from_numpy(np.rint((self.x[i].clip(-1,1)+1)*127.5).astype(np.uint8))


def evaluate_images(folder):
    folder=Path(folder);path=folder/'image_metrics.json'
    if path.exists():return
    cfg=json.loads((folder/'config.json').read_text());dataset=cfg['dataset']
    train_classifier(dataset)
    with np.load(folder/'samples.npz') as z:x=z['generated']
    with np.load(DATA/(dataset+'.npz')) as z:y=z['test']
    ef=RESULTS/'evaluators'/dataset;model=Classifier(x.shape[1]).cuda();model.load_state_dict(torch.load(ef/'classifier.pt',weights_only=True)['model'])
    logits,features=extract(model,x);pred=logits.argmax(-1);counts=np.bincount(pred,minlength=10)
    out=dict(generated_count=len(x),reference_count=len(y),reference_set='official test split',clipped_pixel_fraction=float(((x<-1)|(x>1)).mean()),class_probabilities=(counts/len(x)).tolist(),class_coverage=int((counts>=.01*len(x)).sum()),classifier=json.loads((ef/'config.json').read_text()),preprocessing='all methods: clip [-1,1], map to [0,255], round uint8 for Inception; normalized float for digit CNN')
    if dataset=='mnist':
        with np.load(ef/'reference_features.npz') as z:out['FD_MNIST']=frechet(features,z['features'])
    else:
        sys.path.insert(0,str(ROOT/'.deps'))
        import torch_fidelity
        torch.hub.set_dir(str(DATA/'weights'))
        out.update(torch_fidelity.calculate_metrics(input1=Uint8Images(x),input2=Uint8Images(y),cuda=True,fid=True,isc=True,kid=False,verbose=False,batch_size=64,rng_seed=93000,cache_root=str(DATA/'metric_cache')))
        out['torch_fidelity_version']=torch_fidelity.__version__
        out['Inception_feature']='inception-v3-compat; 2048 for FID, logits_unbiased for IS; TensorFlow-compatible resize'
        weights=DATA/'weights'/'checkpoints'/'weights-inception-2015-12-05-6726825d.pth'
        out['inception_weights_sha256']=hashlib.sha256(weights.read_bytes()).hexdigest()
    write_json(path,out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',choices=['mnist','cifar10']);p.add_argument('--folder');p.add_argument('--all',action='store_true');a=p.parse_args();torch.set_num_threads(2);threadpool_limits(2)
    if a.train:train_classifier(a.train)
    else:
        for folder in sorted((RESULTS/'runs').glob('*')) if a.all else [Path(a.folder)]:
            if (folder/'metrics.json').exists() and json.loads((folder/'config.json').read_text())['dataset'] in ['mnist','cifar10']:evaluate_images(folder)
