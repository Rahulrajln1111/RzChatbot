import os
from typing import List, Dict
 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
 
CHROMA_DIR = ".data/chroma"
EMBED_MODEL = "nomic-embed-text"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 6
CANDIDATES = 20
 
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
 
embeddings = OllamaEmbeddings(model=EMBED_MODEL)
reranker = CrossEncoder(RERANK_MODEL)
 
 
def collection_name(thread_id: str) -> str:
    return f"thread_{thread_id.replace('-', '_')}"
 
 
def vector_store(thread_id: str) -> Chroma:
    return Chroma(
        collection_name=collection_name(thread_id),
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )
 

def load_document(path: str) -> List[Document]:
    
    ext = os.path.splitext(path)[1].lower()
    docs: List[Document] = []
 
    if ext == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            docs.append(Document(page_content=text, metadata={"source": path, "page": i}))
 
    elif ext == ".docx":
        import docx
        d = docx.Document(path)
        text = "\n".join(p.text for p in d.paragraphs)
        docs.append(Document(page_content=text, metadata={"source": path, "page": 1}))
 
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            docs.append(Document(page_content=f.read(), metadata={"source": path, "page": 1}))
 
    return docs
 
 
def ingest(thread_id: str, paths: List[str]) -> int:
    
    all_docs: List[Document] = []
    for p in paths:
        for d in load_document(p):
            all_docs.extend(splitter.split_documents([d]))
 
    if not all_docs:
        return 0
 
    vs = vector_store(thread_id)
    vs.add_documents(all_docs)
    return len(all_docs)
 
 
def list_sources(thread_id: str) -> List[str]:
    vs = vector_store(thread_id)
    try:
        metas = vs.get()["metadatas"] or []
    except Exception:
        return []
    return sorted({m.get("source", "?") for m in metas})
 
 
 
def dense_retriever(thread_id: str, k: int):
    return vector_store(thread_id).as_retriever(
        search_type="mmr",# Maximal marginal Relevance
        search_kwargs={"k": k, "fetch_k": k * 4, "lambda_mult": 0.6},
    )
 
 
def bm25_retriever(thread_id: str, k: int):
    vs = vector_store(thread_id)
    data = vs.get()
    docs = [
        Document(page_content=t, metadata=m)
        for t, m in zip(data.get("documents", []), data.get("metadatas", []))
    ]
    if not docs:
        return None
    r = BM25Retriever.from_documents(docs)
    r.k = k
    return r
 
 
def retrieve(thread_id: str, query: str, k: int = TOP_K) -> List[Dict]:
    
    dense = dense_retriever(thread_id, CANDIDATES)
    bm25 = bm25_retriever(thread_id, CANDIDATES)
 
    if bm25 is None:
        candidates = dense.invoke(query)
    else:
        ensemble = EnsembleRetriever(
            retrievers=[bm25, dense],
            weights=[0.4, 0.6],                 # tune per corpus
        )
        candidates = ensemble.invoke(query)
 
    if not candidates:
        return []
 
    # Cross-encoder rerank
    pairs = [(query, d.page_content) for d in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
 
    out = []
    for doc, score in ranked[:k]:
        out.append({
            "text": doc.page_content,
            "source": os.path.basename(doc.metadata.get("source", "?")),
            "page": doc.metadata.get("page", "?"),
            "score": float(score),
        })
    return out
 
 
def format_context(results: List[Dict]) -> str:
    if not results:
        return "No relevant passages found in the uploaded documents."
    blocks = []
    for r in results:
        blocks.append(
            f"[source: {r['source']}, page {r['page']}, score {r['score']:.2f}]\n{r['text']}"
        )
    return "\n\n---\n\n".join(blocks)
