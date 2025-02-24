import streamlit as st
import torch
import os
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_core.documents import Document

# 🔹 Load Ngrok URL from Streamlit secrets
if "ngrok" in st.secrets and "url" in st.secrets["ngrok"]:
    OLLAMA_SERVER_URL = st.secrets["ngrok"]["url"]
else:
    st.error("⚠️ Ngrok URL is missing! Please update it in Streamlit secrets.")
    st.stop()

# 🔹 Force GPU usage for Ollama (if available)
os.environ["OLLAMA_CUDA"] = "1" if torch.cuda.is_available() else "0"

# 🔹 Ensure CUDA availability for Torch
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device}")

# 🔹 Chat assistant prompt template
prompt_template = """
You are an AI assistant for answering questions based on provided PDF content. 
Use the following retrieved context to answer concisely in three sentences maximum.
If the answer isn't in the context, say "I don't know."

Question: {question} 
Context: {context} 
Answer:
"""

# 🔹 Directory to store uploaded PDFs
pdf_storage_dir = "uploaded_pdfs/"
os.makedirs(pdf_storage_dir, exist_ok=True)

# 🔹 Initialize Ollama with the Ngrok URL
try:
    embeddings_model = OllamaEmbeddings(model="deepseek-r1:7b", base_url=OLLAMA_SERVER_URL)
    vector_store = InMemoryVectorStore(embedding=embeddings_model)  # 🔹 Fixed initialization
    qa_model = OllamaLLM(model="deepseek-r1:7b", base_url=OLLAMA_SERVER_URL)
except Exception as e:
    st.error(f"⚠️ Error connecting to Ollama server: {e}")
    st.stop()

def save_uploaded_file(uploaded_file):
    """Save the uploaded PDF file to storage."""
    file_path = os.path.join(pdf_storage_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_content(pdf_path):
    """Load text from a PDF using PDFPlumber."""
    try:
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()
        return [doc.page_content for doc in documents]  # 🔹 Extract text properly
    except Exception as e:
        st.error(f"⚠️ Error reading PDF: {e}")
        return []

def split_into_chunks(texts):
    """Convert raw text into LangChain Document objects and split into chunks."""
    documents = [Document(page_content=text) for text in texts]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(documents)

def index_pdf_documents(chunks):
    """Index documents into the vector store for retrieval."""
    vector_store.add_documents(chunks)

def retrieve_relevant_chunks(query):
    """Retrieve top similar chunks related to the user's question."""
    return vector_store.similarity_search(query, k=3)  # 🔹 Fetch top 3 relevant chunks

def generate_answer(question, relevant_chunks):
    """Generate an AI response based on retrieved PDF content."""
    if not relevant_chunks:
        return "I couldn't find relevant information in the document."

    context = "\n\n".join([doc.page_content for doc in relevant_chunks])
    prompt = ChatPromptTemplate.from_template(prompt_template)
    response_chain = prompt | qa_model
    response = response_chain.invoke({"question": question, "context": context})
    
    return response["text"] if isinstance(response, dict) else response  # 🔹 Ensure clean output

# 🔹 Streamlit UI
st.title("📄 Chat with Your PDF")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=False)

if uploaded_file:
    # Save and process PDF
    pdf_path = save_uploaded_file(uploaded_file)
    docs = load_pdf_content(pdf_path)

    if docs:
        doc_chunks = split_into_chunks(docs)
        index_pdf_documents(doc_chunks)

        user_query = st.chat_input("Ask a question about the PDF...")

        if user_query:
            st.chat_message("user").write(user_query)
            matched_chunks = retrieve_relevant_chunks(user_query)
            response = generate_answer(user_query, matched_chunks)
            st.chat_message("assistant").write(response)
    else:
        st.error("⚠️ No text extracted from PDF. Please upload a valid document.")
