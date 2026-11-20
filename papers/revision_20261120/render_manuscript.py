"""Render the Markdown manuscript into HTML, self-contained Word, and PDF.

Run with the project Python: python render_manuscript.py [path/to/manuscript.md].
Uses installed Markdown, BeautifulSoup, lxml, and headless LibreOffice.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse, unquote
from zipfile import ZipFile, ZIP_DEFLATED
import argparse
import html
import subprocess

import markdown
from bs4 import BeautifulSoup
from lxml import etree as ET

CSS = '''
@page {size:A4; margin:18mm;}
body {font-family:"Liberation Serif",serif;font-size:10.5pt;line-height:1.18;color:#111;}
h1 {font-size:18pt;line-height:1.08;}
h2 {font-size:13pt;page-break-after:avoid;}
h3 {font-size:11pt;page-break-after:avoid;}
p {orphans:3;widows:3;}
table {width:100%;border-collapse:collapse;font-size:8pt;page-break-inside:avoid;margin:7px 0;}
th,td {border:1px solid #aaa;padding:4px;vertical-align:middle;}
th {background:#edf2f7;}
tr {page-break-inside:avoid;}
thead {display:table-header-group;}
img {max-width:100%;height:auto;}
.caption {page-break-after:avoid;}
'''
W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
R = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
CT = 'http://schemas.openxmlformats.org/package/2006/content-types'


def tag(name):
    return '{' + W + '}' + name


def keep(paragraph):
    prop = paragraph.find(tag('pPr'))
    if prop is None:
        prop = ET.Element(tag('pPr'))
        paragraph.insert(0, prop)
    if prop.find(tag('keepNext')) is None:
        ET.SubElement(prop, tag('keepNext'))


def finalize_docx(path):
    with ZipFile(path) as archive:
        files = {n: archive.read(n) for n in archive.namelist()}
    rel = ET.fromstring(files['word/_rels/document.xml.rels'])
    types = ET.fromstring(files['[Content_Types].xml'])
    embedded = []
    for item in rel:
        if item.get('Type', '').endswith('/image') and item.get('TargetMode') == 'External':
            target_url = urlparse(item.get('Target'))
            if target_url.scheme not in ('file', ''):
                raise ValueError('Only local figure assets are supported')
            source = Path(unquote(target_url.path))
            if not source.is_absolute():
                source = path.parent / source
            source = source.resolve()
            if source.parent != path.parent.resolve():
                raise ValueError(f'Figure must be alongside manuscript: {source}')
            target = 'media/' + source.name
            files['word/' + target] = source.read_bytes()
            item.set('Target', target)
            del item.attrib['TargetMode']
            embedded.append(item.get('Id'))
            extension = source.suffix[1:].lower()
            if not any(x.get('Extension') == extension for x in types):
                mime = {'png': 'image/png', 'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'svg': 'image/svg+xml'}[extension]
                ET.SubElement(types, '{' + CT + '}Default', Extension=extension, ContentType=mime)
    doc = ET.fromstring(files['word/document.xml'])
    for node in doc.iter():
        key = '{' + R + '}link'
        if node.get(key) in embedded:
            node.set('{' + R + '}embed', node.attrib.pop(key))
    # HTML import can preserve the bitmap's native size despite CSS width.
    drawing_ns = 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing'
    art_ns = 'http://schemas.openxmlformats.org/drawingml/2006/main'
    for drawing in doc.findall('.//' + tag('drawing')):
        extents = drawing.findall('.//{' + drawing_ns + '}extent') + drawing.findall('.//{' + art_ns + '}ext')
        if extents:
            cx, cy = int(extents[0].get('cx')), int(extents[0].get('cy'))
            scale = min(1.0, 161 * 36000 / cx, 108 * 36000 / cy)
            for extent in extents:
                extent.set('cx', str(round(cx * scale)))
                extent.set('cy', str(round(cy * scale)))
    page_size = doc.find('.//' + tag('pgSz'))
    page_margin = doc.find('.//' + tag('pgMar'))
    text_width = int(page_size.get(tag('w'))) - int(page_margin.get(tag('left'))) - int(page_margin.get(tag('right')))
    for table in doc.findall('.//' + tag('tbl')):
        columns = table.findall('./' + tag('tblGrid') + '/' + tag('gridCol'))
        original_width = sum(int(col.get(tag('w'))) for col in columns)
        if original_width > text_width:
            factor = text_width / original_width
            for col in columns:
                col.set(tag('w'), str(round(int(col.get(tag('w'))) * factor)))
            for width in table.findall('.//' + tag('tcW')) + table.findall('./' + tag('tblPr') + '/' + tag('tblW')):
                if width.get(tag('type')) == 'dxa':
                    width.set(tag('w'), str(round(int(width.get(tag('w'))) * factor)))
        for para in table.findall('.//' + tag('p')):
            prop = para.find(tag('pPr'))
            if prop is None:
                prop = ET.Element(tag('pPr'))
                para.insert(0, prop)
            spacing = prop.find(tag('spacing'))
            if spacing is None:
                spacing = ET.SubElement(prop, tag('spacing'))
            spacing.set(tag('before'), '20')
            spacing.set(tag('after'), '40')
        for run in table.findall('.//' + tag('r')):
            prop = run.find(tag('rPr'))
            if prop is None:
                prop = ET.Element(tag('rPr'))
                run.insert(0, prop)
            for key in ('sz', 'szCs'):
                size = prop.find(tag(key))
                if size is None:
                    size = ET.SubElement(prop, tag(key))
                size.set(tag('val'), '17')
        rows = table.findall(tag('tr'))
        for i, row in enumerate(rows):
            trpr = row.find(tag('trPr'))
            if trpr is None:
                trpr = ET.Element(tag('trPr'))
                row.insert(0, trpr)
            if trpr.find(tag('cantSplit')) is None:
                ET.SubElement(trpr, tag('cantSplit'))
            if i == 0 and trpr.find(tag('tblHeader')) is None:
                ET.SubElement(trpr, tag('tblHeader'))
            if i < len(rows) - 1:
                for para in row.findall('.//' + tag('p')):
                    keep(para)
    for para in doc.findall('.//' + tag('p')):
        text = ''.join(para.itertext())
        if text.startswith('Table '):
            keep(para)
        if para.find('.//{http://schemas.openxmlformats.org/drawingml/2006/main}blip') is not None:
            keep(para)
    for filename, node in [('word/document.xml', doc), ('word/_rels/document.xml.rels', rel), ('[Content_Types].xml', types)]:
        files[filename] = ET.tostring(node, xml_declaration=True, encoding='UTF-8', standalone=True)
    temp = path.with_suffix('.tmp.docx')
    with ZipFile(temp, 'w', ZIP_DEFLATED) as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    temp.replace(path)
    return len([x for x in rel if x.get('Type', '').endswith('/image')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?', type=Path, default=Path(__file__).with_name('manuscript.md'))
    args = parser.parse_args()
    source = args.source.resolve()
    out = source.parent
    soup = BeautifulSoup(markdown.markdown(source.read_text(), extensions=['tables']), 'html.parser')
    for table in soup.find_all('table'):
        table['border'] = '1'
        table['cellpadding'] = '4'
        table['cellspacing'] = '0'
        table['style'] = 'page-break-inside:avoid;width:100%;font-size:8pt'
    for para in soup.find_all('p'):
        if para.get_text().startswith('Table '):
            para['class'] = 'caption'
    for image in soup.find_all('img'):
        local = out / unquote(image['src'])
        if not local.is_file():
            raise FileNotFoundError(local)
        image['width'] = '610'
        image['style'] = 'width:161mm;height:auto'
        if image.parent.name == 'p':
            image.parent['style'] = 'page-break-after:avoid;text-align:center'
    title = soup.h1.get_text() if soup.h1 else source.stem
    html_path = source.with_suffix('.html')
    html_path.write_text('<!DOCTYPE html><html><head><meta charset="utf-8"><title>' + html.escape(title) + '</title><style>' + CSS + '</style></head><body>' + str(soup) + '</body></html>')
    docx_path = source.with_suffix('.docx')
    pdf_path = source.with_suffix('.pdf')
    with TemporaryDirectory(prefix='qc-hgt-render-') as temp:
        profile = (Path(temp) / 'lo-profile').as_uri()
        command = ['libreoffice', '-env:UserInstallation=' + profile, '--headless']
        for target in (docx_path, pdf_path):
            if target.exists():
                target.unlink()
        subprocess.run(command + ['--convert-to', 'docx:Office Open XML Text', '--outdir', str(out), str(html_path)], check=True, timeout=120)
        if not docx_path.exists():
            raise RuntimeError('LibreOffice did not create DOCX')
        image_count = finalize_docx(docx_path)
        subprocess.run(command + ['--convert-to', 'pdf:writer_pdf_Export', '--outdir', str(out), str(docx_path)], check=True, timeout=120)
        if not pdf_path.exists():
            raise RuntimeError('LibreOffice did not create PDF')
    print(f'Rendered {html_path.name}, {docx_path.name}, {pdf_path.name}; embedded figures: {image_count}')


if __name__ == '__main__':
    main()
