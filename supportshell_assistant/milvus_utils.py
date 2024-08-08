import logging
from pymilvus import connections, MilvusException
from supportshell_assistant.config import MILVUS_HOST, MILVUS_PORT

logger = logging.getLogger(__name__)

def connect_to_milvus():
    try:
        connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
        logger.info("Connection successful")
    except MilvusException as e:
        logger.error(f"Failed to connect to Milvus: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    return True
