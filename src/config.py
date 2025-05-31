import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
    QDRANT_URL = os.getenv('QDRANT_URL')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
    
    # RAG Settings
    COLLECTION_NAME = "uni_academic_system"
    EMBEDDING_MODEL = "text-embedding-3-small"
    CACHE_SIZE = 100
    
    @classmethod
    def validate(cls):
        required = ['OPENAI_API_KEY', 'DEEPSEEK_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY']
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing: {missing}")