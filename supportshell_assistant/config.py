import os

MILVUS_HOST = os.getenv("MILVUS_HOST", "sushe-milvus.sushe-milvus.svc.cluster.local")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = "logs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gpt2"
# LLM_MODEL = "instructlab/granite-7b-lab"
