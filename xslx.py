import openpyxl
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Alternative for embeddings (avoid GooglePalmEmbeddings):
# Import libraries for text embeddings (e.g., sentence-transformers)
# from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS


def get_text_from_xlsx(xlsx_files):
  """
  Extracts text from a list of XLSX files.

  Args:
      xlsx_files: A list of paths to XLSX files.

  Returns:
      A string containing the combined text from all sheets in all files.
  """
  text = ""
  for xlsx_file in xlsx_files:
    # Open the XLSX file in read mode
    workbook = openpyxl.load_workbook(filename=xlsx_file, read_only=True)
    # Iterate through all sheets in the workbook
    for sheet in workbook.worksheets:
      # Extract text from each cell (modify for specific needs)
      for row in sheet.iter_rows():
        for cell in row:
          text += str(cell.value) + " "  # Add space between cell values
  return text.strip()  # Remove trailing space


def get_text_chunks(text):
  """
  Splits the text into chunks of a specified size with overlap.

  Args:
      text: The entire text string.

  Returns:
      A list of text chunks.
  """
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
  chunks = text_splitter.split_text(text)
  return chunks


def get_vector_store(text_chunks):
  """
  Creates a vector store from the text chunks using alternative embeddings (replace with your choice).

  Args:
      text_chunks: A list of text chunks.

  Returns:
      A FAISS vector store.
  """
