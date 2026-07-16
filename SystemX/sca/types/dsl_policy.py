import ast
from collections.abc import Sequence
from typing import Protocol

class DslSession(Protocol):
    """Per-notebook rewrite state; obtained via DslPolicy.new_session()."""

    def pre_scan(self, sources: list[str]) -> None: ...
    def rewrite(self, tree: ast.Module) -> ast.Module: ...

class DslPolicy(Protocol):
    """Strategy for intercepting operator-overloading DSLs before graph construction."""

    def new_session(self) -> DslSession: ...

class _IdentitySession:
    def pre_scan(self, sources: list[str]) -> None:
        pass

    def rewrite(self, tree: ast.Module) -> ast.Module:
        return tree

class NoDslPolicy:
    """Default policy: the AST is passed through untouched."""

    def new_session(self) -> DslSession:
        return _IdentitySession()

class DslRewritePolicy:
    """Rewrites detected DSL constructs (Beam, Airflow, pipe libraries by default) into method-chain form before the dataflow visitor runs."""

    def __init__(self, plugins: Sequence | None = None) -> None:
        self._plugins = plugins

    def new_session(self) -> DslSession:
        from SystemX.sca.dsl.rewriter import DslRewriter

        return DslRewriter(self._plugins)
