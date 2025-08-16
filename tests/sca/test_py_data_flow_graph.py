import unittest
from unittest.mock import MagicMock, patch, call
import ast
import networkx as nx
from ApexDAG.sca.py_data_flow_graph import PythonDataFlowGraph
from ApexDAG.state import Stack, State
from ApexDAG.util.draw import Draw
from ApexDAG.sca.constants import NODE_TYPES, EDGE_TYPES

class TestPythonDataFlowGraph(unittest.TestCase):

    def setUp(self):
        self.mock_logger = MagicMock()
        patcher = patch('ApexDAG.util.logging.setup_logging', return_value=self.mock_logger)
        self.mock_setup_logging = patcher.start()
        self.addCleanup(patcher.stop)

        self.mock_stack = MagicMock(spec=Stack)
        self.mock_state = MagicMock(spec=State)
        self.mock_stack.get_current_state.return_value = self.mock_state

        patcher = patch('ApexDAG.state.Stack', return_value=self.mock_stack)
        self.mock_stack_class = patcher.start()
        self.addCleanup(patcher.stop)

        self.dfg = PythonDataFlowGraph(notebook_path="test_notebook.ipynb")
        self.dfg.code = ""

    def test_initialization(self):
        self.assertFalse(self.dfg._replace_dataflow)
        self.assertEqual(self.dfg._logger, self.mock_logger)
        self.assertEqual(self.dfg._state_stack, self.mock_stack)
        self.assertEqual(self.dfg._current_state, self.mock_state)
        self.mock_setup_logging.assert_called_once_with("py_data_flow_graph test_notebook.ipynb", False)

    def test_visit_import(self):
        node = ast.Import(names=[ast.alias(name='os', asname=None), ast.alias(name='sys', asname='system')])
        self.dfg.visit_Import(node)
        self.mock_stack.imported_names.__setitem__.assert_has_calls([
            call('os', 'os'),
            call('system', 'sys')
        ])

    def test_visit_import_from(self):
        node = ast.ImportFrom(module='collections', names=[ast.alias(name='defaultdict', asname=None)])
        self.dfg.visit_ImportFrom(node)
        self.mock_stack.imported_names.__setitem__.assert_called_once_with('defaultdict', 'collections')
        self.mock_stack.import_from_modules.__setitem__.assert_called_once_with('defaultdict', 'collections')

    @patch.object(PythonDataFlowGraph, '_get_names', return_value=['x'])
    @patch.object(PythonDataFlowGraph, '_get_versioned_name', return_value='x_1')
    @patch('ApexDAG.sca.py_data_flow_graph.flatten_list', return_value=['x'])
    def test_visit_assign_variable(self, mock_flatten_list, mock_get_versioned_name, mock_get_names):
        node = ast.Assign(targets=[ast.Name(id='x', ctx=ast.Store())], value=ast.Constant(value=1)); node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1
        self.dfg.code = "x = 1"
        
        self.mock_state.variable_versions = {'x': []}
        self.dfg.visit_Assign(node)

        self.mock_state.set_current_variable.assert_has_calls([
            call('x_1'),
            call(None)
        ])
        self.mock_state.set_current_target.assert_has_calls([
            call('x'),
            call(None)
        ])
        self.mock_state.add_node.assert_called_once_with('x_1', NODE_TYPES["VARIABLE"])
        self.mock_state.variable_versions.__getitem__('x').append.assert_called_once_with('x_1')

    @patch.object(PythonDataFlowGraph, '_get_base_name', return_value='x')
    @patch.object(PythonDataFlowGraph, '_get_versioned_name', return_value='x_1')
    @patch.object(PythonDataFlowGraph, '_get_last_variable_version', return_value='x_0')
    def test_visit_aug_assign(self, mock_get_last_variable_version, mock_get_versioned_name, mock_get_base_name):
        node = ast.AugAssign(target=ast.Name(id='x', ctx=ast.Store()), op=ast.Add(), value=ast.Constant(value=1)); node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1
        self.dfg.visit_AugAssign(node)

        self.mock_state.set_current_variable.assert_has_calls([
            call('x_1'),
            call(None)
        ])
        self.mock_state.add_node.assert_called_once_with('x_1', NODE_TYPES["VARIABLE"])
        self.mock_state.add_edge.assert_called_once()
        self.mock_state.variable_versions.__setitem__.assert_called_once()

    @patch.object(PythonDataFlowGraph, 'visit_Assign')
    def test_visit_ann_assign(self, mock_visit_assign):
        node = ast.AnnAssign(target=ast.Name(id='x', ctx=ast.Store()), annotation=ast.Name(id='int', ctx=ast.Load()), value=ast.Constant(value=1), simple=1)
        self.dfg.visit_AnnAssign(node)
        mock_visit_assign.assert_called_once_with(node)

    @patch.object(PythonDataFlowGraph, '_get_names', return_value=['my_func'])
    @patch.object(PythonDataFlowGraph, '_get_versioned_name', return_value='my_func_1')
    @patch.object(PythonDataFlowGraph, '_check_resursion', return_value=False)
    @patch.object(PythonDataFlowGraph, '_process_arguments', return_value={'args': ['a'], 'defaults': []})
    def test_visit_function_def(self, mock_process_arguments, mock_check_recursion, mock_get_versioned_name, mock_get_names):
        node = ast.FunctionDef(name='my_func', args=ast.arguments(), body=[], decorator_list=[], returns=None); node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1
        self.dfg.visit_FunctionDef(node)

        self.mock_stack.functions.__setitem__.assert_called_once()
        self.mock_state.add_node.assert_called_once_with('my_func_1', NODE_TYPES["FUNCTION"])
        self.mock_stack.create_child_state.assert_called_once()
        self.mock_stack.restore_state.assert_called_once()

    @patch.object(PythonDataFlowGraph, '_get_caller_object', return_value='pd')
    @patch.object(PythonDataFlowGraph, '_process_library_call')
    @patch.object(PythonDataFlowGraph, '_process_class_call')
    @patch.object(PythonDataFlowGraph, '_process_function_call')
    @patch.object(PythonDataFlowGraph, '_process_builtin_call')
    @patch.object(PythonDataFlowGraph, '_process_method_call')
    def test_visit_call_library(self, mock_process_method_call, mock_process_builtin_call, mock_process_function_call, mock_process_class_call, mock_process_library_call, mock_get_caller_object):
        node = ast.Call(func=ast.Attribute(value=ast.Name(id='pd', ctx=ast.Load()), attr='read_csv', ctx=ast.Load()), args=[], keywords=[])
        self.mock_stack.imported_names = {'pd': 'pandas'}
        self.dfg.visit_Call(node)
        mock_process_library_call.assert_called_once_with(node, 'pd', 'read_csv')

    @patch.object(PythonDataFlowGraph, '_get_lr_values', return_value=('x', 'y'))
    @patch.object(PythonDataFlowGraph, '_get_last_variable_version', side_effect=['x_1', 'y_1'])
    def test_visit_bin_op(self, mock_get_last_variable_version, mock_get_lr_values):
        node = ast.BinOp(left=ast.Name(id='x', ctx=ast.Load()), op=ast.Add(), right=ast.Name(id='y', ctx=ast.Load())); node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1; node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1
        self.mock_state.current_variable = 'z_1'
        self.dfg.visit_BinOp(node)
        self.mock_state.add_edge.assert_called_with('x_1', 'z_1', 'add', EDGE_TYPES["CALLER"], node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)
        self.assertEqual(self.mock_state.add_edge.call_count, 2)

    @patch.object(PythonDataFlowGraph, '_get_last_variable_version', return_value='x_0')
    @patch.object(PythonDataFlowGraph, '_process_subscript', return_value=('filter', EDGE_TYPES["CALLER"]))
    @patch.object(PythonDataFlowGraph, '_tokenize_method', return_value='x')
    def test_visit_name(self, mock_tokenize_method, mock_process_subscript, mock_get_last_variable_version):
        node = ast.Name(id='x', ctx=ast.Load()); node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1; node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1
        self.mock_state.current_variable = 'y_1'
        self.dfg.visit_Name(node)
        self.mock_state.add_edge.assert_called_once_with(
            'x_0',
            'y_1',
            'x',
            EDGE_TYPES["INPUT"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset
        )

    @patch.object(PythonDataFlowGraph, '_tokenize_method', return_value='attr')
    def test_visit_attribute(self, mock_tokenize_method):
        node = ast.Attribute(value=ast.Name(id='obj', ctx=ast.Load()), attr='attr', ctx=ast.Load()); node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1; node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1
        self.mock_state.last_variable = 'obj_0'
        self.mock_state.current_variable = 'result_0'
        self.dfg.visit_Attribute(node)
        self.mock_state.add_edge.assert_called_once_with(
            'obj_0',
            'result_0',
            'attr',
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset
        )

    @patch.object(PythonDataFlowGraph, '_get_names', return_value=['If'])
    @patch.object(PythonDataFlowGraph, '_get_versioned_name', return_value='If_1')
    @patch.object(PythonDataFlowGraph, '_visit_if_body')
    def test_visit_if(self, mock_visit_if_body, mock_get_versioned_name, mock_get_names):
        node = ast.If(test=ast.Constant(value=True), body=[], orelse=[]); node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1; node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1
        self.dfg.visit_If(node)
        mock_visit_if_body.assert_called_once()
        self.mock_stack.branches.append.assert_called_once()
        self.mock_stack.merge_states.assert_called_once()

    @patch.object(PythonDataFlowGraph, '_get_names', return_value=['Loop'])
    @patch.object(PythonDataFlowGraph, '_get_versioned_name', return_value='Loop_1')
    def test_visit_while(self, mock_get_versioned_name, mock_get_names):
        node = ast.While(test=ast.Constant(value=True), body=[], orelse=[]); node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1; node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1
        self.dfg.visit_While(node)
        self.mock_stack.create_child_state.assert_called_once()
        self.mock_stack.merge_states.assert_called_once()

    @patch.object(PythonDataFlowGraph, '_get_names', return_value=['item'])
    @patch.object(PythonDataFlowGraph, '_get_versioned_name', return_value='item_1')
    @patch.object(PythonDataFlowGraph, 'visit')
    def test_visit_for(self, mock_visit, mock_get_versioned_name, mock_get_names):
        node = ast.For(target=ast.Name(id='item', ctx=ast.Store()), iter=ast.Name(id='iterable', ctx=ast.Load()), body=[], orelse=[]); node.lineno = 1; node.col_offset = 0; node.end_lineno = 1; node.end_col_offset = 1
        self.dfg.visit_For(node)
        self.mock_state.add_node.assert_called_once_with('iterable_None_None', NODE_TYPES["INTERMEDIATE"])
        self.mock_state.add_edge.assert_called_once()
        self.mock_stack.create_child_state.assert_called_once()
        self.mock_stack.merge_states.assert_called_once()

    def test_draw_all_subgraphs(self):
        self.mock_state.variable_versions = {'x': ['x_0'], 'y': ['y_0']}
        with patch.object(self.dfg, 'draw') as mock_draw:
            self.dfg.draw_all_subgraphs()
            mock_draw.assert_has_calls([
                call('x', 'x'),
                call('y', 'y'),
                call()
            ])

    @patch('ApexDAG.sca.py_data_flow_graph.convert_multidigraph_to_digraph', return_value=nx.DiGraph())
    @patch('ApexDAG.util.draw.Draw')
    @patch('ApexDAG.sca.py_data_flow_graph.get_subgraph', return_value=nx.DiGraph())
    def test_draw_with_start_node(self, mock_get_subgraph, mock_draw_class, mock_convert_multidigraph_to_digraph):
        self.mock_state.copy_graph.return_value = nx.DiGraph()
        self.mock_state.get_graph.return_value = nx.MultiDiGraph()
        self.dfg.draw(save_path="test_path", start_node="test_node")
        mock_get_subgraph.assert_called_once()
        mock_convert_multidigraph_to_digraph.assert_called_once()
        mock_draw_class.return_value.dfg.assert_called_once()

    @patch('ApexDAG.sca.py_data_flow_graph.convert_multidigraph_to_digraph', return_value=nx.DiGraph())
    @patch('ApexDAG.util.draw.Draw')
    def test_webrender(self, mock_draw_class, mock_convert_multidigraph_to_digraph):
        self.mock_state.get_graph.return_value = nx.MultiDiGraph()
        self.dfg.webrender(save_path="web_path")
        mock_convert_multidigraph_to_digraph.assert_called_once()
        mock_draw_class.return_value.dfg_webrendering.assert_called_once()

    @patch('ApexDAG.sca.py_data_flow_graph.convert_multidigraph_to_digraph', return_value=nx.DiGraph())
    @patch('ApexDAG.util.draw.Draw')
    def test_to_json(self, mock_draw_class, mock_convert_multidigraph_to_digraph):
        self.mock_state.get_graph.return_value = nx.MultiDiGraph()
        self.dfg.to_json()
        mock_convert_multidigraph_to_digraph.assert_called_once()
        mock_draw_class.return_value.dfg_to_json.assert_called_once()

    @patch('ApexDAG.sca.py_data_flow_graph.convert_multidigraph_to_digraph', return_value=nx.DiGraph())
    @patch('ApexDAG.sca.py_data_flow_graph.save_graph')
    def test_save_dfg(self, mock_save_graph, mock_convert_multidigraph_to_digraph):
        self.mock_state.get_graph.return_value = nx.MultiDiGraph()
        self.dfg.save_dfg(path="save_path")
        mock_convert_multidigraph_to_digraph.assert_called_once()
        mock_save_graph.assert_called_once()

    @patch('ApexDAG.sca.py_data_flow_graph.load_graph', return_value=nx.DiGraph())
    def test_read_dfg(self, mock_load_graph):
        self.dfg.read_dfg(path="read_path")
        mock_load_graph.assert_called_once_with("read_path")
        self.mock_state.set_graph.assert_called_once()

    def test_optimize(self):
        self.dfg.optimize()
        self.mock_state.optimize.assert_called_once()

    def test_filter_relevant(self):
        self.dfg.filter_relevant()
        self.mock_state.filter_relevant.assert_called_once()

    def test_get_graph(self):
        self.dfg.get_graph()
        self.mock_state.get_graph.assert_called_once()

    def test_check_recursion(self):
        # Test case for a recursive function
        code = """
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)
"""
        node = ast.parse(code).body[0]
        self.assertTrue(self.dfg._check_resursion(node))

        # Test case for a non-recursive function
        code = """
def add(a, b):
    return a + b
"""
        node = ast.parse(code).body[0]
        self.assertFalse(self.dfg._check_resursion(node))

    def test_process_arguments(self):
        args = ast.arguments(args=[ast.arg(arg='a'), ast.arg(arg='b')], vararg=None, kwonlyargs=[], kw_defaults=[], defaults=[ast.Constant(value=1)])
        result = self.dfg._process_arguments(args)
        self.assertEqual(result['args'], ['a', 'b'])
        self.assertEqual(len(result['defaults']), 1)

    @patch.object(PythonDataFlowGraph, 'visit')
    def test_visit_if_body(self, mock_visit):
        body = [ast.Expr(value=ast.Constant(value=1))]
        self.dfg._visit_if_body(body, "test_context", "parent_context")
        self.mock_stack.create_child_state.assert_called_once_with("test_context", "parent_context")
        mock_visit.assert_called_once_with(body[0])

    @patch.object(PythonDataFlowGraph, '_get_last_variable_version', return_value='var_0')
    @patch.object(PythonDataFlowGraph, '_tokenize_method', return_value='method')
    @patch.object(PythonDataFlowGraph, '_get_names', return_value=['arg_name'])
    def test_process_method_call(self, mock_get_names, mock_tokenize_method, mock_get_last_variable_version):
        node = MagicMock(spec=ast.Call)
        node.args = [ast.Name(id='arg', ctx=ast.Load())]
        self.mock_state.current_variable = 'result_0'
        self.dfg._process_method_call(node, 'caller', 'tokens')
        self.mock_state.add_edge.assert_called_with(
            'var_0',
            'result_0',
            'method',
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset
        )
        self.assertEqual(self.mock_state.add_edge.call_count, 2)

    @patch.object(PythonDataFlowGraph, '_tokenize_method', return_value='lib_call')
    @patch.object(PythonDataFlowGraph, '_get_names', return_value=['arg_name'])
    @patch.object(PythonDataFlowGraph, '_get_last_variable_version', return_value='arg_0')
    def test_process_library_call(self, mock_get_last_variable_version, mock_get_names, mock_tokenize_method):
        node = MagicMock(spec=ast.Call)
        node.func = MagicMock(spec=ast.Attribute)
        node.args = [ast.Name(id='arg', ctx=ast.Load())]
        self.mock_state.current_variable = 'result_0'
        self.mock_stack.imported_names = {'lib': 'library'}
        self.dfg._process_library_call(node, 'lib', 'tokens')
        self.mock_state.add_node.assert_called_once_with('library', NODE_TYPES["IMPORT"])
        self.mock_state.add_edge.assert_called_with(
            'library',
            'result_0',
            'lib_call',
            EDGE_TYPES["FUNCTION_CALL"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset
        )
        self.assertEqual(self.mock_state.add_edge.call_count, 2)

    @patch.object(PythonDataFlowGraph, '_tokenize_method', return_value='class_call')
    @patch.object(PythonDataFlowGraph, '_get_names', return_value=['arg_name'])
    @patch.object(PythonDataFlowGraph, '_get_last_variable_version', return_value='arg_0')
    def test_process_class_call(self, mock_get_last_variable_version, mock_get_names, mock_tokenize_method):
        node = MagicMock(spec=ast.Call)
        node.func = MagicMock(spec=ast.Attribute)
        node.args = [ast.Name(id='arg', ctx=ast.Load())]
        self.mock_state.current_variable = 'result_0'
        self.mock_stack.classes = {'MyClass': ['MyClass']}
        self.dfg._process_class_call(node, 'MyClass', 'tokens')
        self.mock_state.add_node.assert_called_once_with('MyClass', NODE_TYPES["CLASS"])
        self.mock_state.add_edge.assert_called_with(
            'MyClass',
            'result_0',
            'class_call',
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset
        )
        self.assertEqual(self.mock_state.add_edge.call_count, 2)

    @patch.object(PythonDataFlowGraph, '_get_names', return_value=['arg_name'])
    @patch.object(PythonDataFlowGraph, '_get_last_variable_version', return_value='arg_0')
    @patch('ApexDAG.sca.py_data_flow_graph.flatten_list', return_value=['arg_name'])
    def test_process_builtin_call(self, mock_flatten_list, mock_get_last_variable_version, mock_get_names):
        node = MagicMock(spec=ast.Call)
        node.args = [ast.Name(id='arg', ctx=ast.Load())]
        self.mock_state.current_variable = 'result_0'
        self.dfg._process_builtin_call(node, 'print')
        self.mock_state.add_node.assert_called_once_with('__builtins__', NODE_TYPES["IMPORT"])
        self.mock_state.add_edge.assert_called_with(
            '__builtins__',
            'result_0',
            'print',
            EDGE_TYPES["CALLER"],
            node.lineno,
            node.col_offset,
            node.end_lineno,
            node.end_col_offset
        )
        self.assertEqual(self.mock_state.add_edge.call_count, 2)

    @patch.object(PythonDataFlowGraph, '_tokenize_method', return_value='lib_attr')
    def test_process_library_attr(self, mock_tokenize_method):
        node = MagicMock(spec=ast.Attribute)
        self.mock_state.current_variable = 'result_0'
        self.mock_stack.imported_names = {'lib': 'library'}
        self.dfg._process_library_attr(node, 'lib')
        self.mock_state.add_node.assert_called_once_with('library', NODE_TYPES["IMPORT"])
        self.mock_state.add_edge.assert_called_once()
        self.mock_state.set_last_variable.assert_called_once_with('library')

    def test_process_subscript_compare(self):
        node = MagicMock(spec=ast.AST)
        node.parent = MagicMock(spec=ast.Subscript)
        node.parent.slice = MagicMock(spec=ast.Compare)
        code_segment, edge_type = self.dfg._process_subscript(node)
        self.assertEqual(code_segment, "filter")
        self.assertEqual(edge_type, EDGE_TYPES["CALLER"])

    @patch.object(PythonDataFlowGraph, '_get_names', return_value=['obj'])
    def test_get_caller_object(self, mock_get_names):
        self.mock_stack.imported_names = {'obj': 'object'}
        result = self.dfg._get_caller_object(MagicMock(spec=ast.Name))
        self.assertEqual(result, 'obj')

    def test_get_versioned_name(self):
        result = self.dfg._get_versioned_name('var', 10)
        self.assertEqual(result, 'var_10')

    def test_get_base_name(self):
        node = ast.Name(id='x', ctx=ast.Load())
        self.assertEqual(self.dfg._get_base_name(node), 'x')

        node = ast.Constant(value=1)
        self.assertEqual(self.dfg._get_base_name(node), 1)

        node = ast.Attribute(value=ast.Name(id='obj', ctx=ast.Load()), attr='attr', ctx=ast.Load())
        self.assertEqual(self.dfg._get_base_name(node), 'obj')

    @patch('ApexDAG.sca.py_data_flow_graph.flatten_list', return_value=['x'])
    def test_get_names_name(self, mock_flatten_list):
        node = ast.Name(id='x', ctx=ast.Load())
        self.assertEqual(self.dfg._get_names(node), ['x'])

    def test_tokenize_method(self):
        self.mock_stack.imported_names = {'pd': 'pandas'}
        self.mock_stack.import_from_modules = {}
        result = self.dfg._tokenize_method("pd.read_csv")
        self.assertEqual(result, "read csv")

        result = self.dfg._tokenize_method("my_function_name")
        self.assertEqual(result, "my function name")

        result = self.dfg._tokenize_method("camelCaseMethod")
        self.assertEqual(result, "camel case method")

    @patch.object(State, 'variable_versions', new_callable=MagicMock)
    def test_get_last_variable_version(self, mock_variable_versions):
        mock_variable_versions.__contains__.return_value = True
        mock_variable_versions.__getitem__.return_value = ['x_0', 'x_1']
        self.mock_state.variable_versions = mock_variable_versions
        result = self.dfg._get_last_variable_version('x')
        self.assertEqual(result, 'x_1')

    def test_import_accessible(self):
        self.mock_stack.imported_names = {'os': 'os'}
        self.assertTrue(self.dfg._import_accessible('os'))
        self.assertFalse(self.dfg._import_accessible('non_existent'))

    def test_class_accessible(self):
        self.mock_stack.classes = {'MyClass': ['MyClass']}
        self.assertTrue(self.dfg._class_accessible('MyClass'))
        self.assertFalse(self.dfg._class_accessible('non_existent'))

    def test_function_accessible(self):
        self.mock_stack.functions = {'my_func': {}}
        self.assertTrue(self.dfg._function_accessible('my_func'))
        self.assertFalse(self.dfg._function_accessible('non_existent'))

    @patch.object(PythonDataFlowGraph, '_get_names', side_effect=[['x'], ['y']])
    @patch('ApexDAG.sca.py_data_flow_graph.flatten_list', side_effect=[['x'], ['y']])
    def test_get_lr_values(self, mock_flatten_list, mock_get_names):
        left = MagicMock(spec=ast.Name)
        right = MagicMock(spec=ast.Name)
        left_var, right_var = self.dfg._get_lr_values(left, right)
        self.assertEqual(left_var, 'x')
        self.assertEqual(right_var, 'y')

if __name__ == '__main__':
    unittest.main()
