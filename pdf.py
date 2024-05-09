#pypdf is required package pip install pypdf or include in requirements.txt
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAI
from langchain.vectorstores import FAISS



def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GooglePalmEmbeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


"""
for basic loading a doc using langchain module

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("path of pdf or call a function/api")
pages = loader.load_and_split()

fast parsing the pdf code

from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("example_data/layout-parser-paper.pdf")
data = loader.load()

"""
