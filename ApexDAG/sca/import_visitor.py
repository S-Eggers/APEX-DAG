import ast
from pprint import pformat
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional, Any

class ImportVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports_set: Set[Tuple[str, Optional[str]]] = set()
        self.classes: List[str] = []
        self.functions: List[str] = []
        
        self.import_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.import_counts: Dict[str, int] = defaultdict(int)
        
        self._context_stack: List[str] = []
        self._scope_stack: List[Set[str]] = [set()]

    @property
    def imports(self) -> List[Tuple[str, Optional[str]]]:
        return list(self.imports_set)

    @property
    def current_context(self) -> Optional[str]:
        return ".".join(self._context_stack) if self._context_stack else None

    def _is_shadowed(self, name: str) -> bool:
        for scope in reversed(self._scope_stack[1:]):
            if name in scope:
                return True
        return False

    def _register_target(self, target: ast.AST) -> None:
        if isinstance(target, ast.Name):
            self._scope_stack[-1].add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._register_target(elt)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name: str = alias.asname or alias.name
            self.imports_set.add((alias.name, alias.asname))
            self._scope_stack[0].add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        module: str = node.module or ""
        for alias in node.names:
            name: str = alias.asname or alias.name
            full_path: str = f"{module}.{alias.name}"
            self.imports_set.add((full_path, alias.asname))
            self._scope_stack[0].add(name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.classes.append(node.name)
        self._context_stack.append(node.name)
        self._scope_stack.append(set())
        self.generic_visit(node)
        self._scope_stack.pop()
        self._context_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        full_name: str = f"{self.current_context}.{node.name}" if self.current_context else node.name
        self.functions.append(full_name)
        
        self._context_stack.append(node.name)
        self._scope_stack.append(set())
        
        args = node.args
        all_args = args.args + args.kwonlyargs + getattr(args, "posonlyargs", [])
        for arg in all_args:
            self._scope_stack[-1].add(arg.arg)
        if args.vararg:
            self._scope_stack[-1].add(args.vararg.arg)
        if args.kwarg:
            self._scope_stack[-1].add(args.kwarg.arg)
            
        self.generic_visit(node)
        self._scope_stack.pop()
        self._context_stack.pop()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._register_target(target)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name):
            name: str = node.value.id
            if (name in self.import_usage or name in self._scope_stack[0]) and not self._is_shadowed(name):
                self.import_usage[name][node.attr] += 1
                self.import_counts[name] += 1
                return 
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        name: str = node.id
        if (name in self.import_usage or name in self._scope_stack[0]) and not self._is_shadowed(name):
            self.import_usage[name]["__direct__"] += 1
            self.import_counts[name] += 1
        self.generic_visit(node)