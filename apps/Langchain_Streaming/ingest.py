from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma

from my_api import get_hf_key

HF_TOKEN = get_hf_key()

def persist_dir(file_path):
    data = PyPDFLoader(file_path)
    print("Loading data...")
    content = data.load()
    print("Splitting data...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1024,chunk_overlap=150)
    chunks = splitter.split_documents(content)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5"
    )
    print("Save to db...")
    vectorstore = Chroma.from_documents(chunks, embeddings,persist_directory="./db")
    
if __name__ == "__main__":
    #will change, if you add file upload on streamlit
    data = "./data/Ikigai.pdf" 
    persist_dir(data)
    