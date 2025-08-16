import unittest
from unittest.mock import MagicMock, patch, call
from ApexDAG.vamsa.track_provenance import (
    get_name,
    is_constant,
    drop_traversal,
    list_traversal,
    keyword_traversal,
    iloc_traversal,
    subscript_traversal,
    slice_traversal,
    KBC,
    ProvenanceTracker,
    track_provenance
)

class TestTrackProvenance(unittest.TestCase):

    def test_get_name(self):
        self.assertEqual(get_name("node:id123"), "node")
        self.assertEqual(get_name("node_name"), "node_name")

    def test_is_constant(self):
        prs = [
            ('in1', 'c1', 'op1', 'out1'),
            ('in2', 'c2', 'op2', ['out2a', 'out2b'])
        ]
        self.assertFalse(is_constant('out1', prs))
        self.assertFalse(is_constant('out2a', prs))
        self.assertTrue(is_constant('in1', prs))
        self.assertTrue(is_constant('non_existent', prs))
        self.assertFalse(is_constant(None, prs))

    def test_drop_traversal(self):
        mock_tracker = MagicMock()
        mock_tracker.var_to_pr = {'var1': ['pr1', 'pr2']}
        pr = (['df', 'var1'], 'c', 'drop', 'df_out')
        next_prs = drop_traversal(pr, mock_tracker)
        self.assertEqual(next_prs, ['pr1', 'pr2'])

    def test_list_traversal(self):
        mock_tracker = MagicMock()
        mock_tracker.var_to_pr = {'item1': ['prA']}
        pr = (['list_var', 'item1'], 'c', 'List', 'list_out')
        next_prs = list_traversal(pr, mock_tracker)
        self.assertEqual(next_prs, ['prA'])

    def test_keyword_traversal(self):
        mock_tracker = MagicMock()
        mock_tracker.var_to_pr = {'labels_var': ['prX']}
        pr_labels = (['df', 'labels_var'], 'c', 'keyword', 'labels')
        pr_other = (['df', 'other_var'], 'c', 'keyword', 'other')
        
        self.assertEqual(keyword_traversal(pr_labels, mock_tracker), ['prX'])
        self.assertEqual(keyword_traversal(pr_other, mock_tracker), [])

    def test_iloc_traversal(self):
        mock_tracker = MagicMock()
        mock_tracker.cal_to_pr = {'idx_var': ['pr_iloc']}
        pr = (['df'], 'c', 'iloc', 'idx_var')
        next_prs = iloc_traversal(pr, mock_tracker)
        self.assertEqual(next_prs, ['pr_iloc'])

    def test_subscript_traversal(self):
        mock_tracker = MagicMock()
        mock_tracker.var_to_pr = {'key_var': ['pr_sub']}
        pr = (['df', 'key_var'], 'c', 'Subscript', 'df_out')
        next_prs = subscript_traversal(pr, mock_tracker)
        self.assertEqual(next_prs, ['pr_sub'])

    def test_slice_traversal(self):
        mock_tracker = MagicMock()
        mock_tracker.var_to_pr = {'lower_bound': ['pr_slice']}
        pr = (['seq', 'lower_bound', 'upper_bound'], 'c', 'Slice', 'sliced_seq')
        next_prs = slice_traversal(pr, mock_tracker)
        self.assertEqual(next_prs, ['pr_slice'])

    def test_kbc_structure(self):
        self.assertIn('drop', KBC)
        self.assertIn('traversal_rule', KBC['drop'])
        self.assertIn('column_exclusion', KBC['drop'])
        self.assertEqual(KBC['drop']['traversal_rule'], drop_traversal)

    def test_provenance_tracker_init(self):
        mock_wir = MagicMock()
        prs = [
            ('in1', 'c1', 'op1', 'out1'),
            ('in2', None, 'op2', 'out2')
        ]
        tracker = ProvenanceTracker(mock_wir, prs)
        self.assertEqual(tracker.wir, mock_wir)
        self.assertEqual(tracker.prs, prs)
        self.assertIn('out1', tracker.var_to_pr)
        self.assertIn('c1', tracker.cal_to_pr)
        self.assertEqual(tracker.C_plus, set())
        self.assertEqual(tracker.C_minus, set())
        self.assertEqual(tracker.visited_prs, set())

    @patch.object(ProvenanceTracker, '_guide_eval')
    def test_provenance_tracker_track(self, mock_guide_eval):
        mock_wir = MagicMock()
        mock_wir.annotated_wir.nodes = {
            'var_annotated': {'annotations': ['features']},
            'var_not_annotated': {'annotations': ['other']}
        }
        prs = [
            (['var_annotated'], 'c1', 'op1', 'out1'),
            (['var_not_annotated'], 'c2', 'op2', 'out2')
        ]
        tracker = ProvenanceTracker(mock_wir, prs)
        tracker.track(what_track={'features'})

        mock_guide_eval.assert_called_once_with(prs[0])
        self.assertEqual(tracker.C_plus, set())
        self.assertEqual(tracker.C_minus, set())

    @patch('ApexDAG.vamsa.track_provenance.is_constant', side_effect=lambda var, prs: var == 'const_var')
    def test_provenance_tracker_guide_eval_constant_input(self, mock_is_constant):
        mock_wir = MagicMock()
        prs = []
        tracker = ProvenanceTracker(mock_wir, prs)
        tracker.kbc = {'op_name': {'column_exclusion': False, 'traversal_rule': MagicMock(return_value=[])}}
        pr = (['const_var'], 'c', 'op_name', 'out')
        tracker._guide_eval(pr)
        self.assertIn('const_var', tracker.C_plus)
        self.assertNotIn('const_var', tracker.C_minus)

    @patch('ApexDAG.vamsa.track_provenance.is_constant', side_effect=lambda var, prs: False)
    def test_provenance_tracker_guide_eval_recursive_call(self, mock_is_constant):
        mock_wir = MagicMock()
        prs = []
        mock_traversal_rule = MagicMock(return_value=[('next_in', 'next_c', 'next_op', 'next_out')])
        tracker = ProvenanceTracker(mock_wir, prs)
        tracker.kbc = {
            'op_name': {'column_exclusion': False, 'traversal_rule': mock_traversal_rule},
            'next_op': {'column_exclusion': False, 'traversal_rule': MagicMock(return_value=[])}
        }
        pr = (['in'], 'c', 'op_name', 'out')
        
        with patch.object(tracker, '_guide_eval') as mock_recursive_guide_eval:
            mock_recursive_guide_eval.side_effect = lambda p, col_excl: None if p == pr else unittest.mock.DEFAULT
            tracker._guide_eval(pr)
            mock_recursive_guide_eval.assert_called_with(pr)
            mock_recursive_guide_eval.assert_called_with(('next_in', 'next_c', 'next_op', 'next_out'), col_excl=False)

    @patch('ApexDAG.vamsa.track_provenance.ProvenanceTracker')
    def test_track_provenance(self, MockProvenanceTracker):
        mock_annotated_wir = MagicMock()
        mock_prs = MagicMock()
        mock_tracker_instance = MockProvenanceTracker.return_value
        mock_tracker_instance.track.return_value = ({'c_plus'}, {'c_minus'})

        c_plus, c_minus = track_provenance(mock_annotated_wir, mock_prs, what_track={'features'})

        MockProvenanceTracker.assert_called_once_with(mock_annotated_wir, mock_prs)
        mock_tracker_instance.track.assert_called_once_with(what_track={'features'})
        self.assertEqual(c_plus, {'c_plus'})
        self.assertEqual(c_minus, {'c_minus'})

if __name__ == '__main__':
    unittest.main()
