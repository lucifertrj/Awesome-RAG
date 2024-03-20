from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma


HF_TOKEN = "<replace-with-your-token>"

def persist_dir(file_path):
    data = PyPDFLoader(file_path)
    print("Loading data...")
    content = data.load()
    print("Splitting data...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=512,chunk_overlap=0)
    chunks = splitter.split_documents(content)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN, model_name="mixedbread-ai/mxbai-embed-large-v1"
    )
    print("Save to db...")
    vectorstore = Chroma.from_documents(chunks, embeddings,persist_directory="./db")
    
if __name__ == "__main__":
    #will change, if you add file upload on streamlit
    data = "./data/gsoc_proposal.pdf" #update your data
    persist_dir(data)