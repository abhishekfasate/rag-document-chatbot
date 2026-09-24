import sys
import os

COLLECTION_NAME = os.environ.get("COLLECTION_NAME","resume")
PERSIST_DIRECTORY = os.environ.get("PERSIST_DIRECTORY","./resume_chroma")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL","all-MiniLM-L6-v2")
FILE_PATH = os.environ.get("FILE_PATH","")
