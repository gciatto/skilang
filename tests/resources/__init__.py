from pathlib import Path


DIR = Path(__file__).parent


def resource_path(name: str | Path) -> Path:
    return DIR / name
