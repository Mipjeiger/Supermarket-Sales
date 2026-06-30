from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional

BASE_DIR = Path(__file__).resolve().parent.parent.parent
print(f"BASE_DIR: {BASE_DIR}")