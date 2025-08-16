import unittest
from unittest.mock import MagicMock, patch, call
from ApexDAG.state.stack import Stack
from ApexDAG.state.state import State

class TestStack(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock()
        patcher = patch('ApexDAG.util.logging.setup_logging', return_value=self.mock_logger)
        self.mock_setup_logging = patcher.start()
        self.addCleanup(patcher.stop)

        # Mock the State class to control its behavior
        self.mock_state_class = MagicMock(spec=State)
        self.mock_state_instance_module = MagicMock(spec=State, context="module", parent_context=None)
        self.mock_state_class.return_value = self.mock_state_instance_module
        patcher = patch('ApexDAG.state.stack.State', new=self.mock_state_class)
        self.mock_state_patch = patcher.start()
        self.addCleanup(patcher.stop)

        self.stack = Stack()

    def test_initialization(self):
        self.assertEqual(self.stack.imported_names, {})
        self.assertEqual(self.stack.import_from_modules, {})
        self.assertEqual(self.stack.classes, {})
        self.assertEqual(self.stack.functions, {})
        self.assertEqual(self.stack.branches, [])
        self.assertEqual(self.stack._current_state, "module")
        self.assertIn("module", self.stack._state)
        self.assertEqual(self.stack._state["module"], self.mock_state_instance_module)
        self.mock_setup_logging.assert_called_once_with("state.Stack", False)
        self.mock_state_class.assert_called_once_with("module")

    def test_contains(self):
        self.assertTrue("module" in self.stack)
        self.assertFalse("non_existent" in self.stack)

    def test_create_child_state_with_parent(self):
        self.mock_state_instance_child = MagicMock(spec=State, context="child_context", parent_context="module")
        self.mock_state_class.side_effect = [self.mock_state_instance_module, self.mock_state_instance_child]

        self.stack.create_child_state("child_context", "module")
        self.assertIn("child_context", self.stack._state)
        self.assertEqual(self.stack._current_state, "child_context")
        self.mock_state_class.assert_called_with("child_context", "module")

    def test_create_child_state_without_parent(self):
        self.mock_state_instance_child = MagicMock(spec=State, context="child_context", parent_context=None)
        self.mock_state_class.side_effect = [self.mock_state_instance_module, self.mock_state_instance_child]

        self.stack.create_child_state("child_context")
        self.assertIn("child_context", self.stack._state)
        self.assertEqual(self.stack._current_state, "child_context")
        self.mock_state_class.assert_called_with("child_context", None)

    def test_create_child_state_already_exists(self):
        with self.assertRaises(ValueError):
            self.stack.create_child_state("module")

    def test_create_child_state_parent_not_exists(self):
        with self.assertRaises(ValueError):
            self.stack.create_child_state("new_context", "non_existent_parent")

    def test_restore_state(self):
        self.mock_state_instance_child = MagicMock(spec=State, context="child_context", parent_context="module")
        self.mock_state_class.side_effect = [self.mock_state_instance_module, self.mock_state_instance_child]
        self.stack.create_child_state("child_context", "module")
        self.stack.restore_state("module")
        self.assertEqual(self.stack._current_state, "module")

    def test_restore_state_not_exists(self):
        with self.assertRaises(ValueError):
            self.stack.restore_state("non_existent")

    def test_restore_parent_state(self):
        self.mock_state_instance_child = MagicMock(spec=State, context="child_context", parent_context="module")
        self.mock_state_class.side_effect = [self.mock_state_instance_module, self.mock_state_instance_child]
        self.stack.create_child_state("child_context", "module")
        self.stack._state["child_context"].parent_context = "module" # Ensure parent_context is set for the mock
        self.stack.restore_parent_state()
        self.assertEqual(self.stack._current_state, "module")

    def test_restore_parent_state_no_parent(self):
        self.mock_state_instance_module.parent_context = None # Ensure no parent for module state
        with self.assertRaises(ValueError):
            self.stack.restore_parent_state()

    def test_merge_states(self):
        mock_state_child1 = MagicMock(spec=State, context="child1", parent_context="module")
        mock_state_child2 = MagicMock(spec=State, context="child2", parent_context="module")
        
        self.mock_state_class.side_effect = [
            self.mock_state_instance_module, 
            mock_state_child1, 
            mock_state_child2
        ]
        self.stack.create_child_state("child1", "module")
        self.stack.create_child_state("child2", "module")

        self.stack._state["child1"] = mock_state_child1
        self.stack._state["child2"] = mock_state_child2

        self.stack.merge_states("module", [(mock_state_child1, "label1", "type1"), (mock_state_child2, "label2", "type2")])

        self.mock_state_instance_module.merge.assert_called_once_with(
            (mock_state_child1, "label1", "type1"),
            (mock_state_child2, "label2", "type2")
        )
        self.assertNotIn("child1", self.stack._state)
        self.assertNotIn("child2", self.stack._state)
        self.assertEqual(self.stack._current_state, "module")

    def test_merge_states_base_not_exists(self):
        with self.assertRaises(ValueError):
            self.stack.merge_states("non_existent", [])

    def test_get_current_state(self):
        self.assertEqual(self.stack.get_current_state(), self.mock_state_instance_module)

if __name__ == '__main__':
    unittest.main()
