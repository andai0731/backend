#pypdf is required package

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("path of pdf or call a function/api")
pages = loader.load_and_split()


"""
fast parsing the pdf code

from langchain_community.document_loaders import PyMuPDFLoader
loader = PyMuPDFLoader("example_data/layout-parser-paper.pdf")
data = loader.load()

"""
