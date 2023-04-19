import os
from dotenv import load_dotenv
def load_env():
    load_dotenv()
UPLOAD_FOLDER = os.getenv('UPLOADS', default='uploads/')
RESOURCES = os.getenv('RESOURCES', default='resources/')