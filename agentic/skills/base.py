from abc import ABC, abstractmethod
from pathlib import Path


class Skill(ABC):
    name: str
    version: str
    description: str

    @abstractmethod
    def run(self, image_path: Path, **kwargs) -> dict:
        ...
