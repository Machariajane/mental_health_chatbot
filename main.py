import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Load username and password from environment variables
username = os.getenv('username')
password = os.getenv('password')

# Construct proxy URLs with the loaded username and password
http_proxy = f"http://{username}:{password}@proxy2:8080"
https_proxy = f"http://{username}:{password}@proxy2:8080"

# Set the HTTP_PROXY and HTTPS_PROXY environment variables
os.environ['HTTP_PROXY'] = http_proxy
os.environ['HTTPS_PROXY'] = https_proxy



import requests
from huggingface_hub import configure_http_backend, get_session

# Create a factory function that returns a Session with configured proxies
def backend_factory() -> requests.Session:
    session = requests.Session()
    session.proxies = {"http": http_proxy, "https": https_proxy}
    session.verify= False
    return session


# Set it as the default session factory
configure_http_backend(backend_factory=backend_factory)

# In practice, this is mostly done internally in `huggingface_hub`
session = get_session()
os.environ['CURL_CA_BUNDLE'] = ''
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.llms import CTransformers
import chainlit as cl
from langchain.memory import ConversationBufferMemory

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cpu'})
# Assuming these are correctly defined in your project
# from prompts_chat_pdf import chat_prompt, CONDENSE_QUESTION_PROMPT

app = FastAPI()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str


chat_history = []


class PDFChatBot:
    def __init__(self):
        self.data_path = os.path.join('data')
        self.db_faiss_path = os.path.join('vectordb', 'db_faiss')

    def create_vector_db(self):
        loader = DirectoryLoader(self.data_path, glob='*.pdf', loader_cls=PyPDFLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        db = FAISS.from_documents(texts, embeddings)
        db.save_local(self.db_faiss_path)

    def load_llm(self):
        llm = CTransformers(
            model="./llama-2-7b-chat.Q4_0.gguf",
            model_type="llama",
            max_new_tokens=2000,
            temperature=0.8
        )
        return llm

    # def conversational_chain(self):
    #     db = FAISS.load_local(self.db_faiss_path, embeddings)
    #     memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    #     conversational_chain = ConversationalRetrievalChain.from_llm(llm=self.load_llm(),
    #                                                                  retriever=db.as_retriever(search_kwargs={"k": 3}),
    #                                                                  verbose=True, memory=memory)
    #     return conversational_chain
    def custom_prompt(self, question, retrieved_idx):
        # Combine the question with the context from documents
        introduction = "You are a mental health chatbot designed to provide support and information."

    # Special directive for sensitive topics
        if "suicide" in question.lower():
            introduction = f"{introduction}\n\nIt seems you are going through a very tough time. I strongly encourage you to talk to someone who can help. Please consider calling this number: +25472 for immediate support."
            return introduction
        
        retrieved_docs = [documents_dict[idx] for idx in retrieved_idx]
        # Extract content from documents
        context = " ".join([doc['content'] for doc in retrieved_docs])

        
        prompt = f"Given the following information: {context}\nQuestion: {question}\nAnswer:"
        return prompt
    
    def conversational_chain(self):
        db = FAISS.load_local(self.db_faiss_path, embeddings)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # Define a custom function to format the prompt
        def custom_format_fn(data):
            question = data["question"]
            retrieved_idx = db.retrieve(question)  # Retrieve relevant documents based on the question
            return self.custom_prompt(question, retrieved_idx)

        conversational_chain = ConversationalRetrievalChain.from_llm(llm=self.load_llm(),
                                                                     retriever=db.as_retriever(search_kwargs={"k": 3}),
                                                                     verbose=True, memory=memory,
                                                                     format_fn=custom_format_fn)  # Pass the custom prompt function
        return conversational_chain


bot = PDFChatBot()
bot.create_vector_db()
conversational_chain = bot.conversational_chain()


@app.post("/chat", response_model=ChatResponse)
def chat_with_bot(request: ChatRequest):
    global chat_history
    try:
        res = conversational_chain({"question": request.message, "chat_history": chat_history})
        answer = res["answer"]
        chat_history.append(answer)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8090)
