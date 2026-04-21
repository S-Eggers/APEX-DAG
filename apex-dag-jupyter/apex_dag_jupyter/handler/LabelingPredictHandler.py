import json
import tornado
from jupyter_server.base.handlers import APIHandler

from ApexDAG.pipeline.labeling_pipeline_factory import LabelingPipelineFactory


class LabelingPredictHandler(APIHandler):
    def initialize(self, model):
        self.model = model

    @tornado.web.authenticated
    def post(self):
        try:
            input_data = self.get_json_body()
            pipeline = LabelingPipelineFactory.create(input_data, self.model)
            analysis_results = pipeline.execute(input_data.get("code", ""))
            self.finish(json.dumps({"success": True, "predictions": analysis_results}))
        except SyntaxError as e:
            self.set_status(400)
            self.finish(json.dumps({"success": False, "message": str(e)}))
        except Exception as e:
            self.set_status(500)
            self.finish(json.dumps({"success": False, "message": "Internal Error"}))