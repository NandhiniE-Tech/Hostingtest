#*************$  code absolutely working $***********### in text loader
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
load_dotenv()
from langchain_community.document_loaders import PyPDFLoader
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
NAMESPACE = os.getenv("NAMESPACE")
#
#pdf_loader = PyPDFLoader(r"C:\Users\NANDHINI\Desktop\MyChatBot\Tent n Trek1.pdf")
pdf_loader = PyPDFLoader(r"C:\Users\NANDHINI\Desktop\MyChatBot\Travel Site.pdf")

print("done with textloader", pdf_loader)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=150)

docs = pdf_loader.load_and_split(text_splitter=text_splitter)

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
from langchain_pinecone import PineconeVectorStore
vectorstore = PineconeVectorStore(
    pinecone_api_key=PINECONE_API_KEY,
    embedding=embeddings,
    index_name=PINECONE_INDEX,
    namespace=NAMESPACE
)
vectorstore.add_documents(docs)
print("done")
#print(embeddings)