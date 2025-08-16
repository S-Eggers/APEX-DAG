import unittest
from unittest.mock import MagicMock, patch, call
import networkx as nx
from ApexDAG.state.state import State
from ApexDAG.sca.constants import EDGE_TYPES, NODE_TYPES, DOMAIN_EDGE_TYPES

class TestState(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock()
        patcher = patch('ApexDAG.util.logging.setup_logging', return_value=self.mock_logger)
        self.mock_setup_logging = patcher.start()
        self.addCleanup(patcher.stop)

        self.state = State(name="test_context", parent_context="parent_context")

    def test_initialization(self):
        self.assertEqual(self.state.context, "test_context")
        self.assertEqual(self.state.parent_context, "parent_context")
        self.assertIsInstance(self.state._G, nx.MultiDiGraph)
        self.assertEqual(self.state.edge_for_current_target, {})
        self.assertEqual(self.state.variable_versions, {})
        self.assertEqual(self.state.imported_names, {})
        self.assertEqual(self.state.import_from_modules, {})
        self.assertEqual(self.state.classes, {})
        self.assertEqual(self.state.functions, {})
        self.assertIsNone(self.state.current_target)
        self.assertIsNone(self.state.current_variable)
        self.assertIsNone(self.state.last_variable)
        self.assertIsNone(self.state.payload)
        self.mock_setup_logging.assert_called_once_with("state.State", False)

    def test_getitem(self):
        self.state.current_variable = "var1"
        self.assertEqual(self.state["current_variable"], "var1")
        with self.assertRaises(ValueError):
            self.state["invalid_attr"]

    def test_setitem(self):
        self.state["current_variable"] = "var2"
        self.assertEqual(self.state.current_variable, "var2")
        with self.assertRaises(ValueError):
            self.state["invalid_attr"] = "value"

    def test_set_current_variable(self):
        self.state.set_current_variable("test_var")
        self.assertEqual(self.state.current_variable, "test_var")

    def test_set_current_target(self):
        self.state.set_current_target("test_target")
        self.assertEqual(self.state.current_target, "test_target")

    def test_set_last_variable(self):
        self.state.set_last_variable("test_last_var")
        self.assertEqual(self.state.last_variable, "test_last_var")

    def test_copy_graph(self):
        self.state._G.add_node('a')
        copied_graph = self.state.copy_graph()
        self.assertIsInstance(copied_graph, nx.MultiDiGraph)
        self.assertFalse(copied_graph is self.state._G) # Ensure it's a copy, not the same object
        self.assertTrue(nx.is_isomorphic(copied_graph, self.state._G))

    def test_set_graph(self):
        new_graph = nx.MultiDiGraph()
        new_graph.add_node('b')
        self.state.set_graph(new_graph)
        self.assertEqual(self.state._G, new_graph)

    def test_get_graph(self):
        self.assertEqual(self.state.get_graph(), self.state._G)

    def test_get_node(self):
        self.state._G.add_node('a', label='node_a')
        node_data = self.state.get_node('a')
        self.assertEqual(node_data['label'], 'node_a')

    def test_remove_node(self):
        self.state._G.add_node('a')
        self.state.remove_node('a')
        self.assertFalse(self.state._G.has_node('a'))

    def test_node_iterator(self):
        self.state._G.add_nodes_from(['a', 'b'])
        nodes = list(self.state.node_iterator())
        self.assertEqual(set(nodes), {'a', 'b'})

    def test_adjacent_node_iterator(self):
        self.state._G.add_edges_from([('a', 'b'), ('a', 'c')])
        adj_nodes = list(self.state.adjacent_node_iterator('a'))
        self.assertEqual(set(adj_nodes), {'b', 'c'})

    def test_predecessor_node_iterator(self):
        self.state._G.add_edges_from([('a', 'c'), ('b', 'c')])
        pred_nodes = list(self.state.predecessor_node_iterator('c'))
        self.assertEqual(set(pred_nodes), {'a', 'b'})

    def test_successor_node_iterator(self):
        self.state._G.add_edges_from([('a', 'b'), ('a', 'c')])
        succ_nodes = list(self.state.successor_node_iterator('a'))
        self.assertEqual(set(succ_nodes), {'b', 'c'})

    def test_add_node(self):
        self.state.add_node('new_node', NODE_TYPES["VARIABLE"])
        self.assertTrue(self.state._G.has_node('new_node'))
        self.assertEqual(self.state._G.nodes['new_node']['node_type'], NODE_TYPES["VARIABLE"])

        self.state.add_node('new_node', NODE_TYPES["VARIABLE"]) # Add existing node
        self.mock_logger.debug.assert_called_with("Node %s already exists in the graph", 'new_node')

    def test_node_degree(self):
        self.state._G.add_edges_from([('a', 'b'), ('c', 'b'), ('b', 'd')])
        degrees = self.state.node_degree('b')
        self.assertEqual(degrees["in"], 1) # out_degree in networkx is in_degree for MultiDiGraph
        self.assertEqual(degrees["out"], 2) # in_degree in networkx is out_degree for MultiDiGraph

    def test_has_edge(self):
        self.state._G.add_edge('a', 'b', key='edge1')
        self.assertTrue(self.state.has_edge('a', 'b'))
        self.assertTrue(self.state.has_edge('a', 'b', key='edge1'))
        self.assertFalse(self.state.has_edge('a', 'b', key='edge2'))
        self.assertFalse(self.state.has_edge('a', 'c'))

    def test_get_edge_iterator(self):
        self.state._G.add_edge('a', 'b', key='edge1', code='code1')
        self.state._G.add_edge('a', 'b', key='edge2', code='code2')
        edges = list(self.state.get_edge_iterator('a', 'b'))
        self.assertEqual(len(edges), 2)
        self.assertIn(('edge1', {'code': 'code1'}), edges)
        self.assertIn(('edge2', {'code': 'code2'}), edges)

    def test_set_edge_data(self):
        self.state._G.add_edge('a', 'b', key='edge1', code='old_code')
        self.state.set_edge_data('a', 'b', 'edge1', code='new_code', new_attr='value')
        self.assertEqual(self.state._G['a']['b']['edge1']['code'], 'new_code')
        self.assertEqual(self.state._G['a']['b']['edge1']['new_attr'], 'value')

    def test_remove_edge(self):
        self.state._G.add_edge('a', 'b', key='edge1')
        self.state.remove_edge('a', 'b', 'edge1')
        self.assertFalse(self.state._G.has_edge('a', 'b', key='edge1'))

    def test_get_edge_data(self):
        self.state._G.add_edge('a', 'b', key='edge1', code='test_code')
        data = self.state.get_edge_data('a', 'b', 'edge1', 'code')
        self.assertEqual(data, 'test_code')

    def test_add_edge_new(self):
        self.state.add_edge('a', 'b', 'code_val', EDGE_TYPES["CALLER"], lineno=1)
        self.assertTrue(self.state._G.has_edge('a', 'b', key='a_b_code_val'))
        edge_data = self.state._G['a']['b']['a_b_code_val']
        self.assertEqual(edge_data['code'], 'code_val')
        self.assertEqual(edge_data['edge_type'], EDGE_TYPES["CALLER"])
        self.assertEqual(edge_data['count'], 1)
        self.assertEqual(edge_data['lineno'], 1)

    def test_add_edge_existing(self):
        self.state.add_edge('a', 'b', 'code_val', EDGE_TYPES["CALLER"], lineno=1)
        self.state.add_edge('a', 'b', 'code_val', EDGE_TYPES["CALLER"], lineno=2)
        edge_data = self.state._G['a']['b']['a_b_code_val']
        self.assertEqual(edge_data['count'], 2)

    def test_add_edge_invalid(self):
        self.state.add_edge(None, 'b', 'code_val', EDGE_TYPES["CALLER"])
        self.mock_logger.debug.assert_called_with("Ignoring edge %s -> %s with code %s", None, 'b', 'code_val')

    @patch('networkx.compose', side_effect=lambda g1, g2: g1)
    def test_merge_branched_variables(self, mock_compose):
        state1 = State("s1")
        state1.variable_versions = {'x': ['x_0']}
        state1._G.add_node('x_0')

        state2 = State("s2")
        state2.variable_versions = {'x': ['x_1']}
        state2._G.add_node('x_1')

        self.state.variable_versions = {'x': ['x_base']}
        self.state._G.add_node('x_base')

        self.state.merge((state1, "if", EDGE_TYPES["BRANCH"]), (state2, "else", EDGE_TYPES["BRANCH"]))

        self.assertIn('x', self.state.variable_versions)
        self.assertIn('branch_x_0_x_1', self.state.variable_versions['x'])
        self.assertTrue(self.state._G.has_node('branch_x_0_x_1'))
        self.assertTrue(self.state._G.has_edge('x_0', 'branch_x_0_x_1'))
        self.assertTrue(self.state._G.has_edge('x_1', 'branch_x_0_x_1'))

    @patch('networkx.compose', side_effect=lambda g1, g2: g1)
    def test_merge_looped_variables(self, mock_compose):
        state1 = State("s1")
        state1.variable_versions = {'y': ['y_0']}
        state1._G.add_node('y_0')

        self.state.variable_versions = {'y': ['y_base']}
        self.state._G.add_node('y_base')

        self.state.merge((state1, "loop", EDGE_TYPES["LOOP"]))

        self.assertIn('y', self.state.variable_versions)
        self.assertIn('loop_y_base_y_0', self.state.variable_versions['y'])
        self.assertTrue(self.state._G.has_node('loop_y_base_y_0'))
        self.assertTrue(self.state._G.has_edge('y_0', 'loop_y_base_y_0'))
        self.assertTrue(self.state._G.has_edge('loop_y_base_y_0', 'y_base'))

    @patch('networkx.weakly_connected_components')
    def test_filter_relevant(self, mock_weakly_connected_components):
        # Setup a graph with two components, one relevant, one not
        self.state._G.add_node('a', label='node_a')
        self.state._G.add_node('b', label='node_b')
        self.state._G.add_node('c', label='node_c')
        self.state._G.add_edge('a', 'b', predicted_label=DOMAIN_EDGE_TYPES["DATA_IMPORT_EXTRACTION"])
        self.state._G.add_edge('b', 'c', predicted_label=0) # Not special

        # Mock weakly_connected_components to return two components
        mock_weakly_connected_components.return_value = [{'a', 'b', 'c'}, {'d', 'e'}]

        # Add a non-relevant component
        self.state._G.add_node('d')
        self.state._G.add_node('e')
        self.state._G.add_edge('d', 'e', predicted_label=0)

        self.state.filter_relevant()

        self.assertTrue(self.state._G.has_node('a'))
        self.assertTrue(self.state._G.has_node('b'))
        self.assertTrue(self.state._G.has_node('c'))
        self.assertFalse(self.state._G.has_node('d'))
        self.assertFalse(self.state._G.has_node('e'))

    @patch.object(State, 'node_iterator')
    @patch.object(State, 'node_degree')
    @patch.object(State, 'get_node')
    @patch.object(State, 'successor_node_iterator')
    @patch.object(State, 'predecessor_node_iterator')
    @patch.object(State, 'get_edge_iterator')
    @patch.object(State, 'remove_node')
    @patch.object(State, 'remove_edge')
    @patch.object(State, 'add_edge')
    def test_optimize(self, mock_add_edge, mock_remove_edge, mock_remove_node, mock_get_edge_iterator, mock_predecessor_node_iterator, mock_successor_node_iterator, mock_get_node, mock_node_degree, mock_node_iterator):
        # Test case 1: Isolated node removal
        mock_node_iterator.return_value = ['isolated_node', 'if_node', 'redundant_edge_node_x']
        mock_node_degree.side_effect = [{'in': 0, 'out': 0}, {'in': 1, 'out': 1}, {'in': 1, 'out': 1}]
        mock_get_node.return_value = {'node_type': NODE_TYPES["VARIABLE"]}

        # Test case 2: IF node optimization
        mock_get_node.side_effect = [{'node_type': NODE_TYPES["VARIABLE"]}, {'node_type': NODE_TYPES["IF"]}, {'node_type': NODE_TYPES["VARIABLE"]}]
        mock_successor_node_iterator.return_value = ['next_node']
        mock_predecessor_node_iterator.return_value = ['prev_node']
        mock_get_edge_iterator.return_value = [('key', {'code': 'if_code', 'edge_type': 0})]

        # Test case 3: Redundant edge removal
        mock_get_edge_iterator.side_effect = [[('key1', {'code': 'redundant_edge_node_x', 'edge_type': EDGE_TYPES["INPUT"]}), ('key2', {'code': 'other_code', 'edge_type': 0})]]

        self.state.optimize()

        mock_remove_node.assert_called_once_with('isolated_node')
        # Assertions for IF node optimization
        mock_remove_node.assert_any_call('if_node')
        mock_add_edge.assert_any_call('prev_node', 'next_node', 'if_code', 0)
        # Assertions for redundant edge removal
        mock_remove_edge.assert_called_once_with('redundant_edge_node_x', 'next_node', 'key1')

if __name__ == '__main__':
    unittest.main()
