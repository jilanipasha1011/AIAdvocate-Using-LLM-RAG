import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Path to your PDF folder
DATA_PATH = "data/"
VECTORSTORE_PATH = "faiss_index"

def load_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vectorstore saved to {VECTORSTORE_PATH}")

if __name__ == "__main__":
    print("Loading PDFs...")
    docs = load_documents()
    print(f"Loaded {len(docs)} pages.")
    print("Splitting documents...")
    chunks = split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    print("Creating vectorstore...")
    create_vectorstore(chunks)