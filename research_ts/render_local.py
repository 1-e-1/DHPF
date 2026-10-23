"""Canonical documents renderer, local LibreOffice + PyMuPDF raster adapter.

Uses the skill's conversion and page naming flow. Only the unavailable Poppler
PDF raster step is replaced with the existing PyMuPDF package.
"""
import importlib.util,json,os,sys,subprocess
from pathlib import Path
import pymupdf
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tmp/paper_final_render'
SKILL=Path(__import__('os').environ.get('DOCUMENTS_SKILL_DIR', str(Path(__file__).resolve().parents[1] / 'optional_tools/documents')))
def main():
    os.environ['PATH']=str(ROOT/'tmp/lo_portable/extracted/LibreOffice/program')+os.pathsep+os.environ['PATH']
    spec=importlib.util.spec_from_file_location('canonical_docx_renderer',SKILL/'render_docx.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    def raster(pdf_path,dpi,fmt,thread_count,output_folder,paths_only,output_file):
        result=[]
        with pymupdf.open(pdf_path) as doc:
            for i,page in enumerate(doc):
                p=Path(output_folder)/f'page0001-{i+1:02d}.png';page.get_pixmap(matrix=pymupdf.Matrix(dpi/72,dpi/72),alpha=False).save(p);result.append(str(p))
        return result
    mod.convert_from_path=raster
    # LibreOffice's profile expects a correctly escaped Windows file URI.
    def run(cmd,env=None,verbose=False):
        cmd=[('-env:UserInstallation='+Path(c.split('file://',1)[1]).as_uri()) if c.startswith('-env:UserInstallation=file://') else c for c in cmd]
        return subprocess.run(cmd,env=env,capture_output=True,text=True,encoding='utf-8',errors='replace',check=False)
    mod._run_cmd=run
    source=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'DHPF_时序数据生成研究论文.docx'
    output=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else OUT
    pages=mod.rasterize(str(source),str(output),120,True,True)
    pdf=next(output.glob('*.pdf'))
    with pymupdf.open(pdf) as doc:
        for i,p in enumerate(doc):(output/f'page-{i+1}.txt').write_text(p.get_text(),encoding='utf-8')
        records=[dict(page=i+1,characters=len(p.get_text()),size=list(p.rect)) for i,p in enumerate(doc)]
    (output/'render_metadata.json').write_text(json.dumps(dict(renderer='documents/render_docx.py + officially verified local LibreOffice 25.2.7.2; PyMuPDF replaces Poppler raster only',pages=records),indent=2),encoding='utf-8')
    print('Rendered',len(pages),'pages')
if __name__=='__main__':main()
