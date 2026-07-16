import csv
from dataclasses import asdict
from pathlib import Path

from .domain import LineageTuple
from .interfaces import TupleWriterPolicy


class CSVTupleWriter(TupleWriterPolicy):
    def write(self, tuples: list[LineageTuple], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        target_file = output_path.with_suffix(".csv")

        with open(target_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["tuple_type", "subject_id", "object_id"])
            writer.writeheader()
            for t in tuples:
                writer.writerow(asdict(t))
