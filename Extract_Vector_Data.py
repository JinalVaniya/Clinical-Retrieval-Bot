import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="pydantic"
)

from langchain_community.document_loaders import PyMuPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()

# Load data
DATA_PATH = "Data/"
def load_file(data):
    loader = DirectoryLoader(data,
                             glob = "*.pdf",
                             loader_cls= PyMuPDFLoader)
    documents = loader.load()
    return documents

documents= load_file(data = DATA_PATH)
print("Length of PDF: ", len(documents))

# Create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                                   chunk_overlap = 50)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(extracted_data=documents)
print("Length of Text Chunks: ",len(text_chunks))

# Create vector embeddings
def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2",
                                            cache_folder=".\hf_cache")
    return embedding_model

embedding_model = get_embedding_model()

# Store embeddings in FAISS
DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
