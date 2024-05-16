import psycopg2  # Assuming you're using PostgreSQL
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as palm
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain_google_genai import GoogleGenerativeAI


def get_pdf_text(pdf_docs):  # pdf_docs = [".pdf", ".pdf", ...] (200 MB - at max in one iteration)
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


def store_embeddings_in_pgvector(text_chunks, pg_connection_string):
    embeddings = GooglePalmEmbeddings(google_api_key=API)

    # Connect to PostgreSQL database
    conn = psycopg2.connect(pg_connection_string)
    cur = conn.cursor()

    # Create table to store embeddings (if not already exists)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            text TEXT,
            embedding vector(128)  -- Assuming 128-dimensional embeddings
        );
    """)

    # Insert each text chunk and its embedding into the table
    for chunk in text_chunks:
        embedding = embeddings.encode_text(chunk)
        cur.execute("""
            INSERT INTO embeddings (text, embedding)
            VALUES (%s, %s);
        """, (chunk, embedding))

    # Commit changes and close connection
    conn.commit()
    cur.close()
    conn.close()


def main():
    pdf_docs = [".pdf", ".pdf", ...]  # Replace with your list of PDF files
    text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(text)

    # Replace with your PostgreSQL connection string
    pg_connection_string = "host=localhost dbname=your_database user=your_user password=your_password"
    store_embeddings_in_pgvector(text_chunks, pg_connection_string)


"""
from langchain_cohere import CohereEmbeddings
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

connection = "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"  # Uses psycopg3!
collection_name = "my_docs"
embeddings = CohereEmbeddings()

vectorstore = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
)

"""


if __name__ == "__main__":
    main()
