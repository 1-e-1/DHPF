"""Explicit Chinese academic fonts without inherited Office theme overrides."""
from pathlib import Path
import shutil
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from lxml import etree

def fonts(pr, east='宋体', latin='Times New Roman'):
    el=pr.find(qn('w:rFonts'))
    if el is None:
        el=OxmlElement('w:rFonts');pr.insert(0,el)
    el.attrib.clear()
    for key,value in [('ascii',latin),('hAnsi',latin),('cs',latin),('eastAsia',east),('hint','eastAsia')]:
        el.set(qn('w:'+key),value)
    lang=pr.find(qn('w:lang'))
    if lang is None:lang=OxmlElement('w:lang');pr.append(lang)
    lang.set(qn('w:val'),'en-US');lang.set(qn('w:eastAsia'),'zh-CN')

def normalize(doc):
    for style in doc.styles:
        east='黑体' if style.style_id.startswith(('Title','Heading')) else '宋体'
        fonts(style.element.get_or_add_rPr(),east)
    for pr in doc.styles.element.xpath('./w:docDefaults/w:rPrDefault/w:rPr'):
        fonts(pr)
    roots=[doc._element]
    for section in doc.sections:
        roots.extend([section.header._element,section.footer._element])
    for root in roots:
        for p in root.xpath('.//w:p'):
            ids=p.xpath('./w:pPr/w:pStyle/@w:val')
            east='黑体' if ids and ids[0].startswith(('Title','Heading')) else '宋体'
            for run in p.xpath('.//w:r'):
                # Native OMML mathematics is intentionally kept intact.
                if any(e.tag==qn('m:oMath') for e in run.iterancestors()):continue
                fonts(run.get_or_add_rPr(),east)
    ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
    for rel in doc.part.rels.values():
        if rel.reltype.endswith('/theme'):
            part=rel.target_part;root=etree.fromstring(part.blob)
            for group in root.xpath('.//a:majorFont | .//a:minorFont',namespaces=ns):
                east='黑体' if etree.QName(group).localname=='majorFont' else '宋体'
                for node in group.xpath('./a:latin',namespaces=ns):node.set('typeface','Times New Roman')
                for node in group.xpath('./a:ea | ./a:font[@script="Hans"]',namespaces=ns):node.set('typeface',east)
            part._blob=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)

def main():
    root=Path(__file__).resolve().parents[1]
    path=root/'DHPF_时序数据生成研究论文.docx'
    backup=root/'tmp/font_revision/original.docx';backup.parent.mkdir(parents=True,exist_ok=True)
    if not backup.exists():shutil.copy2(path,backup)
    doc=Document(path)
    original=[e.text for e in doc._element.xpath('.//w:t | .//m:t')]
    math=[etree.tostring(e) for e in doc._element.xpath('.//m:oMath')]
    normalize(doc)
    assert original==[e.text for e in doc._element.xpath('.//w:t | .//m:t')]
    assert math==[etree.tostring(e) for e in doc._element.xpath('.//m:oMath')]
    doc.save(path)
    print('Fonts updated; all text and native equations unchanged.')

if __name__=='__main__':main()
