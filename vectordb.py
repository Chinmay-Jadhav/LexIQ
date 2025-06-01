from langchain_community.document_loaders import PDFPlumberLoader, PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

#Step 1 : Upload and Load Raw PDF file

pdfs_directory = r"C:/Users/admin/Chinmay/LangChain/PROJECTS/LAWYER/pdfs/"

def upload_pdf(file) :
    with open(pdfs_directory + file.name , "wb") as f :
        f.write(file.getbuffer())

def load_pdf(directory_path) :
    loader = PyPDFDirectoryLoader(
        path=directory_path,
        glob = "**/[!.]*.pdf",

        )
    documents = loader.load()
    return documents

directory_path = r"C:/Users/admin/Chinmay/LangChain/PROJECTS/LAWYER/dataset/"
documents = load_pdf(directory_path)
# print(len(documents))

#Step 2 : Chunking
def create_chunks(documents) :
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True
    )
    text_chunks = text_splitter.split_documents(documents)
    return text_chunks

text_chunks = create_chunks(documents)
# print("Chunks Count : ", len(text_chunks))


#Step 3 : Setup EmbeddingModel (Use Deepseek R1 with Ollama)

ollama_model_name = "mxbai-embed-large:335m"

def get_embedding_model(ollama_model_name) :
    embeddings = OllamaEmbeddings(model = ollama_model_name)
    return embeddings


#Step 4 : Index Documents **Store embeddings in FAISS (vector store)
FAISS_DB_PATH = r"C:/Users/admin/Chinmay/LangChain/PROJECTS/LAWYER/vectorstore/db_faiss"
faiss_db = FAISS.from_documents(text_chunks, get_embedding_model(ollama_model_name))
faiss_db.save_local(FAISS_DB_PATH)