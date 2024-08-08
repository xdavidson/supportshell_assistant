import logging
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_documents(directory_path, glob_pattern="**/*"):
    logger.info("Loading documents from directory...")
    directory_loader = DirectoryLoader(
        path=directory_path,
        glob=glob_pattern,
        silent_errors=True,
        recursive=True,
        use_multithreading=True
    )
    loaded_docs = directory_loader.lazy_load()
    documents = [
        Document(
            page_content=doc.page_content,
            metadata={"source": doc.metadata.get("source", "unknown")},
        )
        for doc in loaded_docs
    ]
    logger.info(f"Loaded {len(documents)} documents.")
    return documents

def remove_duplicates(logs):
    logger.info("Removing duplicate documents...")
    unique_logs = list({doc.page_content: doc for doc in logs}.values())
    logger.info(f"Reduced to {len(unique_logs)} unique documents.")
    return unique_logs

def split_documents(unique_logs, chunk_size=500, chunk_overlap=100):
    logger.info("Splitting documents into smaller chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_documents(unique_logs)
    logger.info(f"Split into {len(chunks)} chunks.")
    return chunks
