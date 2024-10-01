from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import shutil
from get_embedding_function import get_embedding_function

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()
# Change environment variable name from "OPENAI_API_KEY" to the name given in
# your .env file.

CHROMA_PATH = "chroma"
DATA_PATH = "data/pdf_to_md"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=600,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    embedding_function = get_embedding_function()
    # Create a new DB from the documents.
    db = Chroma.from_documents(
        documents=chunks, persist_directory=CHROMA_PATH, embedding=embedding_function,
        collection_name='documents',
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()
