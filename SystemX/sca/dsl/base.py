import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

@dataclass
class ImportContext:
    """Imports collected across all notebook cells before graph construction."""

    aliases: dict[str, str] = field(default_factory=dict)
    from_imports: dict[str, str] = field(default_factory=dict)

    def has_module(self, module: str) -> bool:
        prefix = module + "."
        return any(target == module or target.startswith(prefix) for target in (*self.aliases.values(), *self.from_imports.values()))

    def aliases_for(self, module: str) -> set[str]:
        prefix = module + "."
        return {alias for alias, target in self.aliases.items() if target == module or target.startswith(prefix)}

    def from_import_names(self, module: str) -> set[str]:
        prefix = module + "."
        return {name for name, source in self.from_imports.items() if source == module or source.startswith(prefix)}

class DslPlugin(ABC):
    """One DSL dialect: an import-based activation trigger plus an AST rewrite."""

    name: ClassVar[str]
    trigger_modules: ClassVar[frozenset[str]]

    def is_active(self, imports: ImportContext) -> bool:
        return any(imports.has_module(module) for module in self.trigger_modules)

    @abstractmethod
    def make_transformer(self, imports: ImportContext) -> ast.NodeTransformer: ...
