import logging
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from langchain_milvus import Milvus
from supportshell_assistant.milvus_utils import connect_to_milvus
from supportshell_assistant.document_processing import load_documents, remove_duplicates, split_documents
from supportshell_assistant.config import EMBEDDING_MODEL, LLM_MODEL, MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME

logger = logging.getLogger(__name__)

def initialize_qa_chain():
    if not connect_to_milvus():
        return None

    # Initialize SentenceTransformer model for embeddings
    logger.info("Initializing embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    logger.info(f"Load pretrained SentenceTransformer: {EMBEDDING_MODEL}")

    # Initialize Hugging Face pipeline
    logger.info("Initializing Hugging Face pipeline...")
    llm_pipeline = pipeline("text-generation", model=LLM_MODEL, max_length=1024, truncation=True)
    llm = HuggingFacePipeline(pipeline=llm_pipeline)
    logger.info(f"Loaded Hugging Face pipeline with model: {LLM_MODEL}")

    # Example preprocessing and storing logs
    extracted_logs_path = './data/case_123/sosreport-host0-2024-05-17-chmroof'
    glob_pattern = "var/log/messages"

    try:
        logs = load_documents(extracted_logs_path, glob_pattern)
        unique_logs = remove_duplicates(logs)
        split_docs = split_documents(unique_logs)

        vector_store = Milvus.from_documents(
            documents=split_docs,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            connection_args={
                "host": MILVUS_HOST,
                "port": MILVUS_PORT
            },
            drop_old=True,  # Drop the old Milvus collection if it exists
        )
        logger.info("Vector store initialized successfully")
    except MilvusException as e:
        logger.error(f"MilvusException during initialization: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Milvus vector store: {e}")
        return None

    context = extracted_logs_path
    system_prompt = (
        "Use the given context to answer the question. "
        "If you don't know the answer, say you don't know. "
        "Extract and highlight any critical failures found in the logs. "
        "Keep the response concise and relevant. Context: {context}. "
    )

    my_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{input}"),
        ]
    )
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=my_prompt)
    return create_retrieval_chain(vector_store.as_retriever(), qa_chain)

def ask_question(chain, question):
    response = chain.invoke({"input": question})
    if isinstance(response, dict):
        answer = response.get("answer", "No answer found.")
        relevant_docs = response.get("context", [])
    else:
        answer = response
        relevant_docs = []

    logger.info(f"Question: {question}")
    logger.info(f"Answer: {answer}\n")
    logger.info("Sources:")

    printed_sources = set()
    if relevant_docs:
        for doc in relevant_docs:
            source = doc.metadata.get("source", "unknown")
            if source not in printed_sources:
                logger.info(doc)
                logger.info(f"Filename: {source}")
                printed_sources.add(source)
    else:
        logger.info("No sources found.")
