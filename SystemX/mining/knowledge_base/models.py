from dataclasses import dataclass

import networkx as nx
from pydantic import BaseModel, Field


class KBEntry(BaseModel):
    Library: str = Field(description="The root library (e.g., 'pandas', 'sklearn')")
    Module: str | None = Field(None, description="Submodule if applicable (e.g., 'model_selection')")
    Caller: str | None = Field(None, description="The calling object context (e.g., 'model', 'data')")
    API_Name: str = Field(description="The exact function or method name (e.g., 'read_csv', 'fit')")
    Inputs: list[str] = Field(description="Semantic input labels (e.g., ['features', 'labels'])")
    Outputs: list[str] = Field(description="Semantic output labels (e.g., ['data', 'model'])")


class BatchKBProposal(BaseModel):
    entries: list[KBEntry] = Field(description="A list of newly proposed Knowledge Base entries.")


@dataclass
class CachedNotebook:
    filepath: str
    base_graph: nx.DiGraph
    prs: list[tuple]
