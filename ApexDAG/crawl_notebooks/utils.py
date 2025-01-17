from dataclasses import dataclass, field


@dataclass
class RepositoryData:
    user: str
    repository: str
    data_files: list[str] = field(default_factory=list)
    notebook_files: list[str] = field(default_factory=list)

class InvalidNotebookException(Exception):
    def __init__(self, message):
        super().__init__(message)