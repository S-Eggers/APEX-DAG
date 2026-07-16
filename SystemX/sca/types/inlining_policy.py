from typing import Protocol

from SystemX.state import Stack


class InliningPolicy(Protocol):
    """Strategy for determining if a call should be inlined."""

    def should_inline_function(self, function_name: str, stack: Stack) -> bool: ...
    def should_inline_class_method(self, class_name: str, method_name: str, stack: Stack) -> bool: ...


class ReplaceDataflowPolicy(InliningPolicy):
    """The UDF inlining policy."""

    def should_inline_function(self, function_name: str, stack: Stack) -> bool:
        return function_name in stack.functions and not stack.functions[function_name]["is_recursive"]

    def should_inline_class_method(self, class_name: str, method_name: str, stack: Stack) -> bool:
        full_method_name = f"{class_name}.{method_name}"
        return class_name in stack.classes and full_method_name in stack.functions and not stack.functions[full_method_name]["is_recursive"]


class NoInliningPolicy(InliningPolicy):
    """Default policy that never inlines."""

    def should_inline_function(self, function_name: str, stack: Stack) -> bool:
        return False

    def should_inline_class_method(self, class_name: str, method_name: str, stack: Stack) -> bool:
        return False
