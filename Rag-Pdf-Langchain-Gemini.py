
"""
RAG with LangChain + Gemini 3.7 Flash
PDF -> Embeddings -> Vector Search -> Gemini -> Formatted Output + Local Tracer

NOTE: No Gemini connection code here - you already have it.
Just pass your existing llm object into build_rag_chain().
"""
import os
import json
import time
import uuid
from pathlib import Path
from datetime import datetime

# ---------- 1. LOCAL FILE TRACER ----------
from langchain_core.callbacks import BaseCallbackHandler

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))

class LocalFileTracer(BaseCallbackHandler):
    def __init__(self, log_dir: Path = LOG_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = str(uuid.uuid4())[:8]
        today = datetime.now().strftime("%Y-%m-%d")
        self.file_path = self.log_dir / f"langchain-{today}.jsonl"
        self._starts = {}

    def _write(self, event, run_id, data: dict):
        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "run_id": str(run_id),
            "event": event,
            **data
        }
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def on_chain_start(self, serialized, inputs, run_id, **kwargs):
        self._starts[str(run_id)] = time.time()
        self._write("chain_start", run_id, {"inputs": str(inputs)[:2000]})

    def on_chain_end(self, outputs, run_id, **kwargs):
        latency = int((time.time() - self._starts.get(str(run_id), time.time())) * 1000)
        self._write("chain_end", run_id, {"latency_ms": latency, "outputs": str(outputs)[:2000]})

    def on_chain_error(self, error, run_id, **kwargs):
        self._write("chain_error", run_id, {"error": str(error)})

    def on_llm_start(self, serialized, prompts, run_id, **kwargs):
        self._starts[str(run_id)] = time.time()
        self._write("llm_start", run_id, {"prompts": [str(p)[:2000] for p in prompts]})

    def on_llm_end(self, response, run_id, **kwargs):
        latency = int((time.time() - self._starts.get(str(run_id), time.time())) * 1000)
        usage = {}
        if getattr(response, "llm_output", None):
            usage = response.llm_output.get("token_usage", {})
            # Gemini uses usage_metadata, capture it if present
            if not usage and getattr(response, "usage_metadata", None):
                usage = str(response.usage_metadata)
        self._write("llm_end", run_id, {"latency_ms": latency, "token_usage": usage})

    def on_retriever_end(self, documents, run_id, **kwargs):
        docs = []
        for d in documents:
            text = d.page_content if hasattr(d, "page_content") else str(d)
            docs.append({"page": d.metadata.get("page", "?"), "text": text[:500]})
        self._write("retriever_end", run_id, {"retrieved_docs": docs})


# ---------- 2. FORMATTED OUTPUT ----------
from pydantic import BaseModel, Field
from typing import List

class RagAnswer(BaseModel):
    answer: str = Field(description="Answer grounded in PDF only")
    source_pages: List[int] = Field(description="Page numbers used, 1-based")
    confidence: str = Field(description="High / Medium / Low")
    retrieved_excerpts: List[str] = Field(description="Exact excerpts used")


# ---------- 3. LANGCHAIN RAG ----------
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join([f"[Page {d.metadata.get('page', 0)+1}] {d.page_content}" for d in docs])


def build_rag_chain(pdf_path: str, llm, tracer: LocalFileTracer, embeddings=None):
    """
    llm: YOUR existing Gemini 3.7 Flash llm object. Pass it in, do not create here.
         Example you already have:
         # llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", ...)
         # chain = build_rag_chain("sample.pdf", llm=llm, tracer=tracer)

    embeddings: Optional. If None, uses local HuggingFace embeddings.
                If you already have Gemini embeddings, pass them here too.
    """
    # 1. Parse PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Parsed {len(documents)} pages")
    tracer._write("pdf_parse", str(uuid.uuid4()), {"pdf": pdf_path, "num_pages": len(documents)})

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    # 3. Embeddings - keep local by default, so only LLM call goes to Gemini
    if embeddings is None:
        print("Using local embeddings: sentence-transformers/all-MiniLM-L6-v2")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    # 4. Vector Store + Search
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    print("Vector store ready (FAISS)")

    # 5. Prompt
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Answer ONLY from the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}
"""
    )

    # 6. Gemini Structured Output - Best for Gemini 3.7 Flash
    # This forces Gemini to return RagAnswer JSON directly, no manual parser needed
    structured_llm = llm.with_structured_output(RagAnswer)

    # 7. LCEL Chain
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | structured_llm
    )
    return chain


def main():
    tracer = LocalFileTracer()
    print(f"Logs -> {tracer.file_path}")

    # --- YOU ALREADY HAVE THIS - DO NOT CREATE AGAIN ---
    # llm = your existing Gemini 3.7 Flash object
    # Example of what you already have (commented, for reference only):
    # llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0)
    llm = None  # <-- Replace this line with your existing llm variable
    if llm is None:
        print("Plug your existing Gemini llm here: llm = your_gemini_llm")
        print("Then call: chain = build_rag_chain('sample_policy.pdf', llm=llm, tracer=tracer)")
        return

    chain = build_rag_chain("sample_policy.pdf", llm=llm, tracer=tracer)

    for q in [
        "What is the refund policy for cancelled orders?",
        "How long does shipping take?",
    ]:
        print(f"\nQUESTION: {q}")
        # Tracer passed via config, logs all steps locally
        result: RagAnswer = chain.invoke(q, config={"callbacks": [tracer]})
        print(f"ANSWER: {result.answer}")
        print(f"SOURCES: Pages {result.source_pages} | Confidence: {result.confidence}")
        print("JSON:", result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
