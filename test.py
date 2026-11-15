import traceback
from rag_core import load_document, summarize_pages

path = input("Full path to your PDF: ").strip('"')
with open(path, "rb") as f:
    pages = load_document(f.read(), "test.pdf")
print(len(pages), "pages loaded")

try:
    print(summarize_pages(pages))
except Exception:
    traceback.print_exc()