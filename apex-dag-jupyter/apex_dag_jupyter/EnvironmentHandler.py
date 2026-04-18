import ast
import json
import tornado
from jupyter_server.base.handlers import APIHandler

from ApexDAG.sca.import_visitor import ImportVisitor
from ApexDAG.sca.complexity_visitor import ComplexityVisitor


class EnvironmentHandler(APIHandler):
    def initialize(self, jupyter_server_app_config=None):
        self.jupyter_server_app_config = jupyter_server_app_config
        self.last_analysis_results = {}

    @tornado.web.authenticated
    def post(self):
        try:
            input_data = self.get_json_body()
            code = input_data.get("code", "")

            if not code.strip():
                self.finish(json.dumps({
                    "success": True,
                    "environment_data": self._empty_payload()
                }))
                return

            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                self.log.error(f"SyntaxError in EnvironmentHandler: {e}")
                self.set_status(400)
                self.finish(json.dumps({
                    "message": "Syntax error in notebook. Returning last valid state.",
                    "success": False,
                    "environment_data": self.last_analysis_results
                }))
                return

            # Execute Visitors
            import_visitor = ImportVisitor()
            complexity_visitor = ComplexityVisitor()
            
            import_visitor.visit(tree)
            complexity_visitor.visit(tree)

            sanitized_usage = {
                module: dict(usages) for module, usages in import_visitor.import_usage.items()
            }

            self.last_analysis_results = {
                "imports": {
                    "declared": import_visitor.imports, # Now deduped
                    "usage": sanitized_usage,
                    "counts": dict(import_visitor.import_counts),
                    "classes_defined": import_visitor.classes,
                    "functions_defined": import_visitor.functions
                },
                "complexity": complexity_visitor.metrics
            }

            self.finish(json.dumps({
                "message": "Environment analyzed successfully.",
                "success": True,
                "environment_data": self.last_analysis_results
            }))

        except Exception as e:
            self.log.error(f"Unexpected error in EnvironmentHandler: {e}", exc_info=True)
            self.set_status(500)
            self.finish(json.dumps({
                "message": "Internal server error.",
                "success": False
            }))

    def _empty_payload(self):
        return {
            "imports": {"declared": [], "usage": {}, "counts": {}, "classes_defined": [], "functions_defined": []},
            "complexity": {"loops": 0, "for_else": 0, "while_else": 0, "branches": 0, "match_cases": 0, "list_comp": 0, "dict_comp": 0, "set_comp": 0, "gen_expr": 0, "try_except": 0, "with_blocks": 0, "max_nesting_depth": 0}
        }