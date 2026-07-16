import argparse

from .extractors import DAGLineageExtractor
from .orchestrator import GoldenDatasetBatchProcessor
from .parsers import CytoscapeGraphParser
from .writers import CSVTupleWriter

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract lineage tuples from Cytoscape JSON graphs.")
    parser.add_argument("--dataset", type=str, required=True, help="The name of the dataset folder inside the ./data/ directory.")
    args = parser.parse_args()

    processor = GoldenDatasetBatchProcessor(parser=CytoscapeGraphParser(), extractor=DAGLineageExtractor(), writer=CSVTupleWriter())

    processor.process_dataset(args.dataset)

if __name__ == "__main__":
    main()
