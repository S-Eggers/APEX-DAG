from SystemX.parser.sanitizer_mixin import IPythonSanitizerMixin
from SystemX.sca.cdf_ir import CDFIntermediateRepresentation
from SystemX.sca.types.dsl_policy import DslRewritePolicy, NoDslPolicy
from SystemX.sca.types.inlining_policy import NoInliningPolicy, ReplaceDataflowPolicy
from SystemX.sca.types.provenance_policy import FixedPointProvenanceEnricher, NoProvenancePolicy


class GraphParser(IPythonSanitizerMixin):
    def __init__(self, replace_dataflow: bool = False, enrich_provenance: bool = False, detect_dsl: bool = False) -> None:
        self.inline_policy = ReplaceDataflowPolicy() if replace_dataflow else NoInliningPolicy()
        self.provenance_policy = FixedPointProvenanceEnricher() if enrich_provenance else NoProvenancePolicy()
        self.dsl_policy = DslRewritePolicy() if detect_dsl else NoDslPolicy()

    def parse(self, code: list) -> CDFIntermediateRepresentation:
        graph = CDFIntermediateRepresentation(inlining_policy=self.inline_policy, provenance_policy=self.provenance_policy, dsl_policy=self.dsl_policy)

        sanitized_code = self.sanitize_ipython_cells(code)

        graph.parse_cells(sanitized_code)
        return graph
