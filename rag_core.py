"""Core RAG logic: load -> chunk -> embed -> store -> retrieve -> generate."""
from __future__ import annotations

import io
from dataclasses import dataclass

import faiss
import pymupdf as fitz  
import numpy as np
from docx import Document
from sentence_transformers import SentenceTransformer

from llm import generate

EMBED_MODEL = "all-MiniLM-L6-v2"
_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def load_document(file_bytes: bytes, filename: str) -> list[dict]:
    """Return a list of {"page": int, "text": str} for pdf/docx/txt files."""
    name = filename.lower()
    pages: list[dict] = []

    if name.endswith(".pdf"):
        with fitz.open(stream=file_bytes, filetype="pdf") as pdf:
            for i, page in enumerate(pdf, start=1):
                text = page.get_text().strip()
                if text:
                    pages.append({"page": i, "text": text})
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(file_bytes))
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        if text:
            pages.append({"page": 1, "text": text})
    elif name.endswith(".txt"):
        text = file_bytes.decode("utf-8", errors="ignore").strip()
        if text:
            pages.append({"page": 1, "text": text})
    else:
        raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")

    if not pages:
        raise ValueError("No text could be extracted (scanned PDF?).")
    return pages


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def chunk_pages(pages: list[dict], source: str, chunk_size: int = 800, overlap: int = 150) -> list[Chunk]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    for p in pages:
        text = p["text"]
        for start in range(0, len(text), step):
            piece = text[start : start + chunk_size].strip()
            if len(piece) > 40: 
                chunks.append(Chunk(piece, source, p["page"]))
            if start + chunk_size >= len(text):
                break
    return chunks


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.index: faiss.IndexFlatIP | None = None

    def add(self, new_chunks: list[Chunk]) -> None:
        if not new_chunks:
            return
        vectors = get_embedder().encode(
            [c.text for c in new_chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.chunks.extend(new_chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        q = get_embedder().encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


QA_SYSTEM = (
    "You answer questions strictly using the provided context. "
    "If the answer is not in the context, reply exactly: "
    "'I could not find this in the document.' Do not use outside knowledge."
)


def answer_question(store: VectorStore, question: str, k: int = 4) -> tuple[str, list[tuple[Chunk, float]]]:
    results = store.search(question, k)
    if not results:
        return "Please upload and process a document first.", []

    context = "\n\n".join(
        f"[Source: {c.source}, page {c.page}]\n{c.text}" for c, _ in results
    )
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    return generate(prompt, system=QA_SYSTEM), results


STYLES = {
    "Bullet points": "Write the summary as 6-10 clear bullet points.",
}


def _blocks(pages: list[dict], max_chars: int = 6000) -> list[str]:
    """Group page text into blocks small enough for the LLM context."""
    blocks, current = [], ""
    for p in pages:
        text = p["text"]
        while len(text) > max_chars:  
            if current:
                blocks.append(current)
                current = ""
            blocks.append(text[:max_chars])
            text = text[max_chars:]
        if len(current) + len(text) > max_chars and current:
            blocks.append(current)
            current = ""
        current += text + "\n"
    if current.strip():
        blocks.append(current)
    return blocks


def summarize_pages(pages: list[dict], style: str = "Bullet points") -> str:
    """Map-reduce summarization: summarize each block, then merge the summaries."""
    instruction = STYLES.get(style, STYLES["Bullet points"])
    blocks = _blocks(pages)

    if len(blocks) == 1:
        return generate(f"{instruction}\n\nText:\n{blocks[0]}")

    partials = [
        generate(f"Summarize the key points of this section in 4-6 sentences.\n\nText:\n{b}")
        for b in blocks
    ]
    combined = "\n\n".join(partials)
    return generate(
        f"These are summaries of consecutive sections of one document. "
        f"{instruction}\n\nSection summaries:\n{combined}"
    )
