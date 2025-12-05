import os
from pathlib import Path

from dotenv import load_dotenv

path = Path(os.getcwd()) / ".env"


def load_env_vars():
    load_dotenv(path, override=False)
