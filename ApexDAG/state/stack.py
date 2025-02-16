from typing import Optional

from ApexDAG.util.logging import setup_logging
from ApexDAG.sca.constants import VERBOSE
from ApexDAG.state.state import State


class Stack:
    def __init__(self) -> None:
        self._logger = setup_logging("state.Stack", VERBOSE)

        self.imported_names = {}
        self.import_from_modules = {}
        self.classes = {}
        self.functions = {}
        self.branches = []

        self._current_state = "module"
        self._state = {"module": State("module")}
        
    def create_child_state(self, context: str = None, parent_context: Optional[str] = None) -> None:
        if context in self._state:
            raise ValueError(f"State {context} already exists")

        if parent_context is not None and parent_context not in self._state:
            raise ValueError(f"Parent state {parent_context} does not exist")

        self._state[context] = State(context, parent_context)
        self._current_state = context

    def restore_state(self, context: str) -> None:
        if context in self._state:
            self._current_state = context
        else:
            raise ValueError(f"No state {context} to restore")

    def restore_parent_state(self) -> None:
        parent_context = self._state[self._current_state].parent_context
        if parent_context not in self._state:
            raise ValueError(f"No parent state {parent_context} to restore")

        self.restore_state(parent_context)
    
    def merge_states(self, base_context: str, *args: tuple[str]) -> None:
        if base_context not in self._state:
            raise ValueError(f"No state {base_context} to merge")

        base_state = self._state[base_context]
        base_state.merge_state(*args)
        for context, _, _ in args:
            if context in self._state:
                del self._state[context]

        self._current_state = base_context
            
    def get_current_state(self) -> State:
        return self._state[self._current_state]