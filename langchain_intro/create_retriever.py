from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# REVIEWS_CSV_PATH and REVIEWS_CHROMA_PATH, which are paths where the raw reviews data is stored and where the vector database will store data
REVIEWS_CSV_PATH = r"D:\AIML\RAG-Chatbot\data\reviews.csv"
REVIEWS_CHROMA_PATH = "chroma_data"

loader = CSVLoader(file_path=REVIEWS_CSV_PATH, source_column="  review")
reviews = loader.load()
model_name = "Qwen/Qwen2.5-1.5B"

reviews_vector_db = Chroma.from_documents(
    reviews, HuggingFaceEmbeddings(), persist_directory=REVIEWS_CHROMA_PATH
)

