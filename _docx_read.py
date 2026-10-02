import sys, zipfile, re
path = sys.argv[1]
with zipfile.ZipFile(path) as z:
    xml = z.read("word/document.xml").decode("utf-8", "ignore")
# saltos de parrafo y de linea
xml = xml.replace("</w:p>", "\n").replace("<w:br/>", "\n")
text = re.sub(r"<[^>]+>", "", xml)
import html
text = html.unescape(text)
print(f"=== {path.split(chr(92))[-1]} ===")
print(f"[caracteres: {len(text)}]")
print(text[:1500])
