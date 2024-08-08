import logging
from supportshell_assistant.qa_chain import initialize_qa_chain, ask_question

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
        level=logging.INFO
    )

def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    qa_chain = initialize_qa_chain()
    if qa_chain:
        while True:
            question = input("SupportShell Assistant> Enter your question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                logger.info("Exiting the Q&A session.")
                break
            try:
                ask_question(qa_chain, question)
            except Exception as e:
                logger.error(f"Error during question answering: {e}")
    else:
        logger.error("Vector store is not initialized. Cannot proceed with QA.")

if __name__ == "__main__":
    main()
