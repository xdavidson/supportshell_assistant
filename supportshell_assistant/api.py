from flask import Flask, request, jsonify
from supportshell_assistant.qa_chain import initialize_qa_chain, ask_question
import logging

app = Flask(__name__)

# Initialize QA Chain
qa_chain = initialize_qa_chain()

@app.route('/query', methods=['POST'])
def query():
    if not qa_chain:
        return jsonify({"error": "Vector store is not initialized. Cannot proceed with QA."}), 500
    
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided."}), 400
    
    question = data['question']
    try:
        response = ask_question(qa_chain, question)
        return jsonify({"answer": response["answer"], "sources": response["sources"]})
    except Exception as e:
        logging.error(f"Error during question answering: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
