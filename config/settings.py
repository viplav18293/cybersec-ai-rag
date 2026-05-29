# config/settings.py
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Settings:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PROJECT_NAME = "CyberSec AI RAG"
    VERSION = "1.0.0"
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"

settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)