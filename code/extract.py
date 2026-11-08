"""Extract the unchanged authority DOCX on every run, using standard-library OOXML."""
from pathlib import Path
import zipfile,xml.etree.ElementTree as ET,json,re,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 f=ROOT/'verify/manuscript_reference.docx'
 if not f.exists():
  for rel,expected in json.loads((ROOT/'verify/EXTRACTED_FIXTURE_HASHES.json').read_text()).items():
   assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected, 'Extracted manuscript fixture changed: '+rel
  print('Using hash-verified task-270 manuscript extracts; source DOCX omitted under PDF-only distribution policy.',flush=True)
  return
 z=zipfile.ZipFile(f);ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};root=ET.fromstring(z.read('word/document.xml'));blocks=[];paragraphs=[];table=0;panel=''
 for el in root.find('w:body',ns):
  if el.tag.endswith('}tbl'):
   rows=[[''.join(t.text or '' for t in cell.findall('.//w:t',ns)) for cell in row.findall('w:tc',ns)] for row in el.findall('w:tr',ns)];blocks.append(dict(block=len(blocks)+1,table=table,panel=panel,rows=rows))
  else:
   text=''.join(t.text or '' for t in el.findall('.//w:t',ns));paragraphs.append(text)
   if re.match(r'^Table \d+\.',text):table=int(re.search(r'\d+',text)[0]);panel=''
   if text.startswith('Panel '):panel=text
 assert sorted(set(b['table'] for b in blocks))==list(range(1,11))
 (ROOT/'verify/manuscript_tables.json').write_text(json.dumps(blocks,ensure_ascii=False,indent=2));(ROOT/'verify/manuscript_text.txt').write_text('\n'.join(paragraphs))
 images=[n for n in z.namelist() if n.startswith('word/media/')]
 for i,name in enumerate(images,1):(ROOT/f'verify/manuscript_image{i}.png').write_bytes(z.read(name))
 (ROOT/'verify/MANUSCRIPT_AUTHORITY.json').write_text(json.dumps(dict(filename=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),physical_table_blocks=len(blocks),manuscript_tables=10,embedded_figure_panels=len(images)),indent=2));print('Authority DOCX extracted: 10 tables,',len(blocks),'physical blocks,',len(images),'figure panels',flush=True)
if __name__=='__main__':main()
