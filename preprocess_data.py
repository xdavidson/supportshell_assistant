# preprocess_data.py
import os
import tarfile
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections, DataType

def extract_archive(archive_path, extract_to='/tmp/extracted_logs'):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    if archive_path.endswith('.tar.xz'):
        with tarfile.open(archive_path, 'r:xz') as archive:
            archive.extractall(path=extract_to)
    else:
        with tarfile.open(archive_path, 'r') as archive:
            archive.extractall(path=extract_to)
    return extract_to

def preprocess_logs(extracted_path):
    logs = []
    for root, _, files in os.walk(extracted_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    logs.append(f.read())
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return logs

archive_path = '/data/case_123/sosreport-host0-2024-05-17-chmroof-obfuscated.tar.xz'
extracted_path = extract_archive(archive_path)
logs = preprocess_logs(extracted_path)

# Vectorization
# Load a pre-trained model from Sentence Transformers
model = SentenceTransformer('all-MiniLM-L6-v2')

# Vectorize the logs
embeddings = model.encode(logs)

# Store Data in a Vector Database
# Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define the schema for the collection
fields = [
    {"name": "embeddings", "type": DataType.FLOAT_VECTOR, "params": {"dim": 384}},
    {"name": "log", "type": DataType.STRING}
]
collection = Collection(name="logs", schema=fields)

# Insert the embeddings and logs
entities = [
    {"name": "embeddings", "values": embeddings.tolist()},
    {"name": "log", "values": logs}
]
collection.insert(entities)
