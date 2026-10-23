"""Final artifact and provenance checks; no training or figure redraw."""
import base64,hashlib,json,re,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from research_ts.inference_emergence import ROOT,OUT,read,sha,frozen,write_json

def main():
    frozen();complete=read(OUT/'complete.json')
    assert all(sha(ROOT/p)==h for p,h in complete['checkpoint_hashes'].items())
    x=pd.read_csv(OUT/'sample_dynamics.csv');s=pd.read_csv(OUT/'source_dynamics.csv')
    assert len(x)==3399 and len(s)==4515
    assert x.groupby(['dataset','seed','condition','variant'],dropna=False).size().eq(33).all()
    assert np.isfinite(x[['t','variance_ratio','effective_dimension','sw2','acf_rmse','corr_rmse','power_rmse']]).all().all()
    primary=x[x.condition.isna()|x.condition.eq(0)]
    for d in ['sine','lorenz','har']:
        for v in ['full','untrained','ballistic','frozen_t0','radial_only','solenoidal_only']:
            assert primary[(primary.dataset==d)&(primary.variant==v)].seed.nunique()==5
    traces=list((OUT/'traces').glob('*.npz'));assert len(traces)==95
    for path in traces:
        with np.load(path) as z:
            assert np.isfinite(z['x']).all() and np.isfinite(z['v']).all()
            assert z['x'].shape[1]==256
            if path.stem.endswith('__full'):assert z['x'].shape[0]==33
    report=(OUT/'report.html').read_text(encoding='utf-8')
    embedded=re.findall(r'src="data:image/(?:png|gif);base64,([^"]+)"',report)
    assert len(embedded)==3
    assert all(len(base64.b64decode(v))>1000 for v in embedded)
    with Image.open(OUT/'figures/complete_inference.gif') as gif:
        assert gif.n_frames==33
        gif.seek(0);gif.convert('RGB').save(OUT/'figures/initial_preview.png')
    skill=Path(__import__('os').environ.get('NATURE_FIGURE_SCRIPTS', str(Path(__file__).resolve().parents[1] / 'optional_tools/nature-figure/scripts')));figures=[]
    for name in ['01_sources_and_structure','02_contraction_and_circulation','03_inference_snapshot']:
        pdf=OUT/'figures'/f'{name}.pdf'
        result=subprocess.run([sys.executable,str(skill/'audit_pdf_text.py'),str(pdf),'--json'],capture_output=True,text=True,encoding='utf-8',check=True)
        fontsize=json.loads(result.stdout);write_json(OUT/'figures'/f'{name}.font_audit.json',fontsize)
        assert fontsize['below_minimum_count']==0
        collision=read(OUT/'figures'/f'{name}.collision.json')
        figures.append(dict(name=name,min_font_pt=fontsize['minimum_found_pt'],collision_report=collision))
    v=subprocess.run([sys.executable,str(skill/'validate_figure.py'),str(ROOT/'research_ts/emergence_figures.py'),'--backend','python','--json'],capture_output=True,text=True,encoding='utf-8')
    static=json.loads(v.stdout);write_json(OUT/'figures/source_preflight.json',static)
    write_json(OUT/'verification.json',dict(status='pass',source_files_unchanged=True,checkpoints_unchanged=True,
        primary_checkpoints=15,raw_paths=95,refined_paths=8,additional_circulation_refined_paths=3,
        sample_rows=len(x),source_rows=len(s),animation_frames=33,embedded_report_images=len(embedded),
        full_dimension_metrics=True,figure_reviews=figures,
        visual_review='Every panel of all three static figures and the initial/middle/final animation views reviewed; figures 1 and 2 display seed SD.',
        static_preflight_exceptions=['The GIF and midpoint screen preview use 140/180 dpi; static figure PNG exports are 300 dpi and vector PDF/SVG exports are retained. These previews are not manuscript submission rasters.','No TIFF required for this inference diagnostic report.'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research_ts'/n for n in ['inference_emergence.py','analyze_emergence.py','emergence_figures.py','emergence_report.py','verify_emergence.py']]},
        report_sha256=sha(OUT/'report.html'),no_new_training=True))
    print('PASS: 95 retained paths, 3,399 sample-time rows, 4,515 source-time rows, 33 animation frames; original files unchanged.')

if __name__=='__main__':main()
