import ast
from pprint import pformat
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Optional

class ImportVisitor(ast.NodeVisitor):
    """
    AST Visitor to track import usage, definitions, and counts.
    Implements a lexical scope stack to prevent false positives when 
    local variables shadow imported modules.
    """

    def __init__(self) -> None:
        self.imports: List[Tuple[str, Optional[str]]] = []
        self.classes: List[str] = []
        self.functions: List[str] = []
        
        self.import_usage: Dict[str, Set[Optional[str]]] = defaultdict(set)
        self.import_counts: Dict[str, int] = defaultdict(int)
        
        # Tracks the fully qualified name context (e.g., "MyClass.my_method")
        self._context_stack: List[str] = []
        
        # Symbol table representation: stack of sets containing local variable names.
        # Index 0 is the global module scope.
        self._scope_stack: List[Set[str]] = [set()]

    @property
    def current_context(self) -> Optional[str]:
        """Returns the current fully qualified execution context."""
        return ".".join(self._context_stack) if self._context_stack else None

    def _is_shadowed(self, name: str) -> bool:
        """
        Checks if a name is shadowed in the current local scopes.
        Ignores the global scope (index 0) where imports reside.
        """
        for scope in reversed(self._scope_stack[1:]):
            if name in scope:
                return True
        return False

    def _register_target(self, target: ast.AST) -> None:
        """Recursively registers assignment targets into the current scope."""
        if isinstance(target, ast.Name):
            self._scope_stack[-1].add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._register_target(elt)

    def visit_Import(self, node: ast.Import) -> None:
        """Records standard imports and initializes their usage tracking."""
        for alias in node.names:
            name: str = alias.asname or alias.name
            self.imports.append((alias.name, alias.asname))
            if name not in self.import_usage:
                self.import_usage[name] = set()
            self._scope_stack[0].add(name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Records from-imports and initializes their usage tracking."""
        module: str = node.module or ""
        for alias in node.names:
            name: str = alias.asname or alias.name
            full_path: str = f"{module}.{alias.name}"
            self.imports.append((full_path, alias.asname))
            if name not in self.import_usage:
                self.import_usage[name] = set()
            self._scope_stack[0].add(name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Pushes a new scope and context for class definitions."""
        self.classes.append(node.name)
        self._context_stack.append(node.name)
        self._scope_stack.append(set())
        
        self.generic_visit(node)
        
        self._scope_stack.pop()
        self._context_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Pushes a new scope and context for function definitions, tracking arguments."""
        full_name: str = f"{self.current_context}.{node.name}" if self.current_context else node.name
        self.functions.append(full_name)
        
        self._context_stack.append(node.name)
        self._scope_stack.append(set())
        
        # Register function arguments in the new local scope
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
        """Registers local variable assignments to detect shadowing."""
        for target in node.targets:
            self._register_target(target)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Tracks attribute access on imported modules, verifying scope."""
        if isinstance(node.value, ast.Name):
            name: str = node.value.id
            if name in self.import_usage and not self._is_shadowed(name):
                self.import_usage[name].add(node.attr)
                self.import_counts[name] += 1
                return  # Prevent double-counting in visit_Name
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        """Tracks direct usage of imported modules/functions, verifying scope."""
        name: str = node.id
        if name in self.import_usage and not self._is_shadowed(name):
            self.import_usage[name].add(None)
            self.import_counts[name] += 1
        self.generic_visit(node)

    def __repr__(self) -> str:
        """
        Returns a multi-line, pretty-printed representation of the AST visitor state.
        """
        # Restructure the internal state into a clean dictionary
        # Cast sets to lists for uniform bracket formatting in the output
        state: Dict[str, Any] = {
            "imports": self.imports,
            "classes": self.classes,
            "functions": self.functions,
            "import_usage": {k: list(v) for k, v in self.import_usage.items()},
            "import_counts": dict(self.import_counts)
        }
        
        # Generate the formatted string with a 4-space indent
        formatted_state: str = pformat(state, indent=4, width=80, sort_dicts=False)
        
        return f"{self.__class__.__name__}(\n{formatted_state}\n)"
