import json
import sys
from pathlib import Path

import tqdm

from .interfaces import GraphParserPolicy, TupleExtractionPolicy, TupleWriterPolicy


class GoldenDatasetBatchProcessor:
    def __init__(self, parser: GraphParserPolicy, extractor: TupleExtractionPolicy, writer: TupleWriterPolicy) -> None:
        self._parser = parser
        self._extractor = extractor
        self._writer = writer

    def process_dataset(self, dataset_name: str, base_dir: str = "./data") -> None:
        base_path = Path(base_dir).resolve()
        input_dir = base_path / dataset_name / "annotations"
        output_dir = base_path / dataset_name / "tuples"

        if not input_dir.exists() or not input_dir.is_dir():
            sys.exit(f"Error: Input directory does not exist -> {input_dir}")

        json_files = list(input_dir.glob("*.json"))
        if not json_files:
            print(f"Warning: No JSON files found in {input_dir}")
            return

        print(f"Initiating batch processing for dataset: {dataset_name} ({len(json_files)} files)")

        progress_bar = tqdm.tqdm(json_files)
        for file_path in progress_bar:
            try:
                with open(file_path, encoding="utf-8") as f:
                    raw_data = json.load(f)

                graph = self._parser.parse(raw_data)
                tuples = self._extractor.extract(graph)

                output_file_path = output_dir / file_path.name
                self._writer.write(tuples, output_file_path)

                progress_bar.write(f"Processed: {file_path.name} -> {len(tuples)} tuples")

            except json.JSONDecodeError:
                progress_bar.write(f"Error decoding JSON in file: {file_path.name}")
            except Exception as e:
                progress_bar.write(f"Unexpected error processing {file_path.name}: {e}")
