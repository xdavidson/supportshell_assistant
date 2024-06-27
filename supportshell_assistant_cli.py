# supportshell_assistant_cli.py
import argparse
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections

def query_milvus(query_text):
    # Connect to Milvus
    connections.connect("default", host="localhost", port="19530")
    collection = Collection("logs")

    # Load Sentence Transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query_text]).tolist()

    # Search in Milvus
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(data=query_embedding, anns_field="embeddings", param=search_params, limit=10)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Query log data")
    parser.add_argument("query", type=str, help="Query text")
    args = parser.parse_args()
    
    results = query_milvus(args.query)
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
