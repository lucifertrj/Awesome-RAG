from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

HF_TOKEN = "<replace-with-your-token>"
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN, model_name="mixedbread-ai/mxbai-embed-large-v1"
)
print("Retrieving...")

# R- Retrieval
db = Chroma(persist_directory="./db", 
            embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k":2})

print("Prompt...")
#A - Augment
template = """
User: You are an AI Assistant that follows instructions extremely well.
Please be truthful and give direct answers. Please tell 'I don't know' if user query is not in CONTEXT

Keep in mind, you will lose the job, if you answer out of CONTEXT questions

CONTEXT: {context}
Query: {question}

Remember only return AI answer
Assistant:
"""
prompt = ChatPromptTemplate.from_template(template)

# G- Generator
llm = Ollama(model="llama2",callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

print("Generating the response....")
output_parser = StrOutputParser()
chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | output_parser
)

print("\nAI response:.... \n")
query = "what is the project goal?"
print(chain.invoke(query))