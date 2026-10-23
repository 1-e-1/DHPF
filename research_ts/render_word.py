"""Local Microsoft Word rendering fallback when LibreOffice is unavailable."""
from pathlib import Path
import json
import pythoncom
import win32com.client
import pymupdf
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'tmp/paper_final_render'
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'DHPF_时序数据生成研究论文.docx';pdf=OUT/'manuscript.pdf'
    pythoncom.CoInitialize();word=None;doc=None
    try:
        word=win32com.client.DispatchEx('Word.Application');word.Visible=False;word.DisplayAlerts=0
        doc=word.Documents.Open(str(source),ReadOnly=True,AddToRecentFiles=False)
        doc.Repaginate();doc.ExportAsFixedFormat(str(pdf),17)
    finally:
        if doc is not None:doc.Close(False)
        if word is not None:word.Quit()
        pythoncom.CoUninitialize()
    with pymupdf.open(pdf) as pdfdoc:
        pages=[]
        for i,page in enumerate(pdfdoc):
            page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False).save(OUT/f'page-{i+1}.png')
            txt=page.get_text();(OUT/f'page-{i+1}.txt').write_text(txt,encoding='utf-8')
            pages.append(dict(page=i+1,characters=len(txt),size=list(page.rect)))
        (OUT/'render_metadata.json').write_text(json.dumps(dict(renderer='Microsoft Word COM ExportAsFixedFormat; canonical render_docx.py unavailable because soffice is not installed',pages=pages),indent=2),encoding='utf-8')
        print('Rendered',len(pdfdoc),'pages using Microsoft Word')
if __name__=='__main__':main()
