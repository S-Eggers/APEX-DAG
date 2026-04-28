import json
from pathlib import Path

import tornado
from jupyter_server.base.handlers import APIHandler


class LabelingFlagHandler(APIHandler):
    @tornado.web.authenticated
    def post(self) -> None:
        try:
            input_data = self.get_json_body()
            filename = input_data.get("filename")
            reason = input_data.get("reason")

            if not filename or not reason:
                self.set_status(400)
                self.finish(
                    json.dumps(
                        {"success": False, "message": "Missing filename or reason."}
                    )
                )
                return

            safe_base_dir = Path.home() / ".apexdag"
            safe_base_dir.mkdir(parents=True, exist_ok=True)

            flags_registry = safe_base_dir / "notebook_flags.json"

            # Load existing flags
            flags_data = {}
            if flags_registry.exists():
                with open(flags_registry, encoding="utf-8") as f:
                    flags_data = json.load(f)

            # Update and save
            flags_data[filename] = {"reason": reason}

            with open(flags_registry, "w", encoding="utf-8") as f:
                json.dump(flags_data, f, indent=2)

            self.finish(
                json.dumps(
                    {"success": True, "message": f"Flagged {filename} as {reason}"}
                )
            )

        except Exception as e:
            self.log.error(f"Flag error: {e}", exc_info=True)
            self.set_status(500)
            self.finish(
                json.dumps({"success": False, "message": "Internal Server Error"})
            )
