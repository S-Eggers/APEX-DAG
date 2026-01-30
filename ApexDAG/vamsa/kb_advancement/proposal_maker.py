import requests
from bs4 import BeautifulSoup
import json
import os
import random
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

from groq import Groq
import instructor
from pydantic import BaseModel, Field

from ApexDAG.vamsa.kb_advancement.KBChangeProposal import KBChangeProposal


# ============================
# ===== Pydantic Models =====
# ============================

class KBEntryDetails(BaseModel):
    library: str
    api_name: str
    inputs: List[str]
    outputs: List[str]
    caller: str
    module: Optional[str] = None
    transformation_type: Optional[str] = None


class ProposalResponse(BaseModel):
    description: str
    rationale: str
    expected_impact: str
    change_type: str = "annotation_entry"
    details: KBEntryDetails


# ============================
# ===== Proposal Maker ======
# ============================

class ProposalMaker:
    def __init__(self, link_to_documentation: str, groq_api_key: Optional[str] = None):
        self.link_to_documentation = link_to_documentation
        self.past_proposals: List[Dict] = []
        self.impact_of_past_proposals: List[Dict] = []

        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set")

        client = Groq(api_key=self.api_key)
        self.client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)
        self.model_name = "moonshotai/kimi-k2-instruct-0905"

    # ============================
    # ===== Traversal Logic ======
    # ============================

    def extract_doc_links(self) -> List[str]:
        """Extract internal documentation links from a Sphinx-based API index."""
        response = requests.get(self.link_to_documentation, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        base_netloc = urlparse(self.link_to_documentation).netloc
        links = set()

        for a in soup.select("a[href]"):
            href = a["href"]

            if href.startswith("#") or href.startswith("http"):
                continue

            full_url = urljoin(self.link_to_documentation, href)

            if urlparse(full_url).netloc != base_netloc:
                continue

            # Focus API-generated pages (sklearn / pandas / catboost)
            if "/generated/" in full_url or "/reference/" in full_url:
                links.add(full_url)

        return sorted(links)

    def scrape_page(self, url: str) -> str:
        """Scrape a single documentation page cleanly."""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        # Remove noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        main = soup.find("div", {"role": "main"}) or soup.body
        text = main.get_text(separator="\n", strip=True)

        # Token safety
        return text[:12000]

    def scrape_entire_docs(self, max_pages: int = 5) -> Dict[str, str]:
        """Traverse and scrape all discovered documentation pages."""
        print("🔍 Discovering documentation links...")
        links = self.extract_doc_links()

        scraped_pages = {}
        #sampled links 
        sampled_links = random.sample(links, min(max_pages, len(links)))
        for i, link in enumerate(sampled_links):
            try:
                print(f"📄 [{i + 1}/{max_pages}] Scraping {link}")
                scraped_pages[link] = self.scrape_page(link)
            except Exception as e:
                print(f"❌ Failed to scrape {link}: {e}")

        return scraped_pages

    # ============================
    # ===== LLM Interaction ======
    # ============================

    def query_llm(
        self,
        scraped_docs: Dict[str, str],
        baseline_data: Optional[Dict] = None,
        KB=None
    ) -> Dict:
        structured_input = "\n\n".join(
            f"=== PAGE: {url} ===\n{content}"
            for url, content in scraped_docs.items()
        )

        prompt = f"""
You are an expert in extracting structured Knowledge Base (KB) entries
for a data provenance tracking system (VAMSA).

KB ENTRY FORMAT:
- library
- api_name
- inputs
- outputs
- caller: one of [data, model, metric]
- module (optional)
- transformation_type (optional)

DOCUMENTATION:
{structured_input}

BASELINE REPORT (annotations searched for but unaccounted for in the KB):
{json.dumps(baseline_data['most_common_unannotated_operations'], indent=2) if baseline_data else "None"}

CURRENT KB:
{json.dumps(KB.knowledge_base.to_dict(orient="records"), indent=2) if KB else "None"}

TASK:
Identify ONE high-value API missing from the KB that improves provenance tracking.
Avoid duplicates. Prefer commonly-used APIs.
"""

        messages = [
            {"role": "system", "content": "You extract structured API metadata from documentation."},
            {"role": "user", "content": prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_model=ProposalResponse,
        )

        return response.dict()

    # ============================
    # ===== Main Entry Point =====
    # ============================

    def __call__(self, KB, baseline_report) -> KBChangeProposal:
        print(f"🚀 Starting traversal from {self.link_to_documentation}")

        try:
            scraped_docs = self.scrape_entire_docs(max_pages=2)

            proposal = self.query_llm(
                scraped_docs=scraped_docs,
                baseline_data=baseline_report,
                KB=KB,
            )
        except Exception as e:
            print(f"❌ Error during proposal generation: {e}")
            scraped_docs = self.scrape_entire_docs(max_pages=1)

            proposal = self.query_llm(
                scraped_docs=scraped_docs,
                baseline_data=baseline_report,
                KB=KB,
            )

        details = proposal["details"]

        kb_proposal = KBChangeProposal(
            library=details["library"],
            module=details.get("module"),
            caller=details["caller"],
            api_name=details["api_name"],
            inputs=details["inputs"],
            outputs=details["outputs"],
            change_type=proposal["change_type"],
            description=proposal["description"],
            rationale=proposal["rationale"],
        )

        self.past_proposals.append(proposal)
        return kb_proposal


# ============================
# ===== Example Usage =======
# ============================

if __name__ == "__main__":
    maker = ProposalMaker(
        "https://scikit-learn.org/stable/api/index.html"
    )

    proposal = maker(KB=None, baseline_report=None)

    print("\n=== Generated Proposal ===")
    print(proposal)
