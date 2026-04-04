# KB Advancement

Automated knowledge base improvement through iterative proposal generation and evaluation.

## Overview

This framework automatically enhances annotation and traversal knowledge bases by:
1. **Scraping** library documentation for API signatures
2. **Generating** KB entry proposals using LLMs (Gemini/Groq Cloud)
3. **Evaluating** each proposal's impact on a corpus of notebooks
4. **Accepting** proposals that improve annotation coverage and hit rate

The system runs iteratively, continuously improving the KB based on real-world usage patterns.

## Requirements

### API Keys
- **Google Gemini API**: Required for LLM-based proposal generation
  - Add `GOOGLE_API_KEY=your-api-key-here` to the `.env` file in the project root

### Python Dependencies
```bash
pip install -r ApexDAG/vamsa/kb_advancement/requirements.txt
```

## Quick Start

```bash
chmod +x ./ApexDAG/vamsa/kb_advancement/run_kb_proposals.sh
./ApexDAG/vamsa/kb_advancement/run_kb_proposals.sh
```

### Custom Configuration
```bash
python ApexDAG/vamsa/kb_advancement/propose_changes.py \
    --corpus /path/to/notebooks \
    --docs https://example.com/docs \
    --output output/my_run \
    --iterations 100
```

## Output

Results are saved to the output directory:
- `final_annotation_kb.csv` - Enhanced annotation KB
- `final_traversal_kb.csv` - Enhanced traversal KB
- `temp_report_baseline.json` - Baseline metrics
- `temp_report_enhanced.json` - Enhanced metrics per iteration

## How It Works

1. **Baseline Evaluation**: Measure current KB performance on corpus
2. **Documentation Scraping**: Extract API info from library docs
3. **Proposal Generation**: LLM proposes KB entries based on:
   - Unannotated operations in the corpus
   - Scraped API signatures
   - Current KB gaps
4. **Impact Evaluation**: Test proposal on corpus
5. **Acceptance Decision**: Keep if it improves coverage/hit rate
6. **Iteration**: Repeat with updated KB

## Architecture

- `propose_changes.py` - Main orchestration and evaluation
- `proposal_maker.py` - LLM-based proposal generation
- `KBChangeProposal.py` - Proposal data model
- `scraper.py` - Documentation scraping utilities
- `../evaluate_kb.py` - KB evaluation framework
