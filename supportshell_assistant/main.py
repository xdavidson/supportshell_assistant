import logging
from supportshell_assistant.qa_chain import initialize_qa_chain, ask_question

def main():
    logging.basicConfig(level=logging.INFO)
    qa_chain = initialize_qa_chain()
    if qa_chain:
        while True:
            question = input("SupportShell Assistant> Enter your question (or type 'exit' to quit): ")
            if question.lower() == 'exit':
                print("Exiting the Q&A session.")
                break
            try:
                response = ask_question(qa_chain, question)
                print(f"Question: {question}")
                print(f"Answer: {response['answer']}\n")
                print("Sources:")
                if response['sources']:
                    for source in response['sources']:
                        print(f"- {source}")
                else:
                    print("No sources found.")
            except Exception as e:
                print(f"Error during question answering: {e}")
    else:
        print("Vector store is not initialized. Cannot proceed with QA.")

if __name__ == "__main__":
    main()
