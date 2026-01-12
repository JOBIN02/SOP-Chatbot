import os
import pytesseract
from pdf2image import convert_from_path
from pypdf import PdfReader

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage

# ----------- FORCE CPU EXECUTION -----------
os.environ["OLLAMA_USE_CUDA"] = "0"

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.join(BASE_DIR, "data", "sample.pdf")

EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# OCR dependencies (Windows paths)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"

print("📄 Loading PDF from:", PDF_PATH)

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

# --------------- PDF INGESTION ---------------
def ingest_pdf():
    documents = []
    reader = PdfReader(PDF_PATH)

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={"page": i + 1}
                )
            )

    # OCR fallback for scanned PDFs
    if len(documents) == 0:
        print("⚠ No selectable text found. Using OCR...")

        images = convert_from_path(
            PDF_PATH,
            dpi=300,
            poppler_path=POPPLER_PATH
        )

        for i, img in enumerate(images):
            text = pytesseract.image_to_string(img, lang="eng")
            if text.strip():
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": i + 1}
                    )
                )

        if len(documents) == 0:
            raise ValueError("❌ OCR failed – no text extracted")

    print(f"✅ Pages loaded: {len(documents)}")
    return documents


# --------------- BUILD RAG PIPELINE ---------------
def build_rag_pipeline():
    raw_docs = ingest_pdf()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    split_docs = splitter.split_documents(raw_docs)

    print(f"✅ Chunks created: {len(split_docs)}")

    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = FAISS.from_documents(split_docs, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatOllama(model=LLM_MODEL, temperature=0)

    return retriever, llm


# Build pipeline once
retriever, llm = build_rag_pipeline()


# --------------- QUERY FUNCTION FOR UI ---------------
def ask_question(question: str) -> str:
    relevant_docs = retriever.invoke(question)

    if not relevant_docs:
        return "Answer not available in the document."

    context = "\n\n".join(
        [f"(Page {d.metadata.get('page')}): {d.page_content}" for d in relevant_docs]
    )

    prompt = f"""
Act as a Supervisor of Business Integrity Services (BIS) to answer all the questions of the outside people"

Context:
{context}

Question:
{question}
"""

    message = HumanMessage(content=prompt)

    response = llm.invoke([message])

    sources = sorted({d.metadata.get("page") for d in relevant_docs})

    final_answer = f"{response.content}\n\n📄 Source pages: {', '.join(map(str, sources))}"

    return final_answer
