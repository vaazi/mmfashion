import os
from dotenv import load_dotenv

load_dotenv()

class ApplicationConfig:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'testing')
    DEBUG = True
    FLASK_DEBUG = "1"
    FLASK_APP = "run.py"
