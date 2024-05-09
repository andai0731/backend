from sentencepiece import SentencePieceProcessor  # Assuming SentencePiece for text processing
from faiss import FAISS, IndexFlatL2  # Import FAISS for vector store
from sentence_transformers import SentenceTransformer, util  # Import for embeddings


def get_text_from_txt(txt_files):
  """
  Extracts text from a list of txt files.

  Args:
      txt_files: A list of paths to txt files.

  Returns:
      A string containing the combined text from all files.
  """
  text = ""
  for txt_file in txt_files:
    with open(txt_file, 'r') as f:
      text += f.read()
  return text

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
  Creates a vector store from the text chunks using GooglePalmEmbeddings.

  Args:
      text_chunks: A list of text chunks.

  Returns:
      A FAISS vector store.
  """
  embeddings = GooglePalmEmbeddings()
  vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
  return vector_store



'''
THIS SECTION IS FR TESTING DIFFERENT METHOD OF SPLITTING THE TEXT AND THUS SHOULD BE USED TO EVALUATE THE RESULTS 
from typing import AsyncIterator, Iterator

from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document


class CustomDocumentLoader(BaseLoader):
    """An example document loader that reads a file line by line."""

    def __init__(self, file_path: str) -> None:
        """Initialize the loader with a file path.

        Args:
            file_path: The path to the file to load.
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]:  # <-- Does not take any arguments
        """A lazy loader that reads a file line by line.

        When you're implementing lazy load methods, you should use a generator
        to yield documents one by one.
        """
        with open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            for line in f:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1

    # alazy_load is OPTIONAL.
    # If you leave out the implementation, a default implementation which delegates to lazy_load will be used!
    async def alazy_load(
        self,
    ) -> AsyncIterator[Document]:  # <-- Does not take any arguments
        """An async lazy loader that reads a file line by line."""
        # Requires aiofiles
        # Install with `pip install aiofiles`
        # https://github.com/Tinche/aiofiles
        import aiofiles

        async with aiofiles.open(self.file_path, encoding="utf-8") as f:
            line_number = 0
            async for line in f:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path},
                )
                line_number += 1
'''
