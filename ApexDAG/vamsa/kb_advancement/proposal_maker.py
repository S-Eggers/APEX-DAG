import requests
from bs4 import BeautifulSoup
import json
from typing import Dict, List, Optional
import os
from groq import Groq
import instructor
from pydantic import BaseModel, Field
from ApexDAG.vamsa.kb_advancement.KBChangeProposal import KBChangeProposal


class KBEntryDetails(BaseModel):
    library: str = Field(description="Library name (e.g., 'pandas', 'sklearn', 'catboost')")
    api_name: str = Field(description="API name (e.g., 'fillna', 'CatBoostClassifier.fit')")
    inputs: List[str] = Field(description="Input types (e.g., ['features'], ['data', 'labels'])")
    outputs: List[str] = Field(description="Output types (e.g., ['features'], ['model'])")
    caller: str = Field(description="Caller type: 'data', 'model', or 'metric'")
    module: Optional[str] = Field(default=None, description="Module name if applicable (e.g., 'model.selection')")
    transformation_type: Optional[str] = Field(default=None, description="Semantic description (e.g., 'imputation', 'training')")

class ProposalResponse(BaseModel):
    description: str = Field(description="Brief description of what API this adds tracking for")
    rationale: str = Field(description="Why this API is important for provenance tracking")
    expected_impact: str = Field(description="Expected improvement in tracking coverage")
    change_type: str = Field(default="annotation_entry", description="Type of change")
    details: KBEntryDetails = Field(description="The KB entry details")
    

class ProposalMaker:
    def __init__(self, link_to_documentation: str, groq_api_key: Optional[str] = None):
        self.past_proposals = []
        self.link_to_documentation = link_to_documentation
        self.impact_of_past_proposals = []
        self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        
        client = Groq(api_key=self.api_key)
        self.client = instructor.from_groq(client, mode=instructor.Mode.TOOLS)
        self.model_name = "moonshotai/kimi-k2-instruct-0905"
    
    def scrape_documentation(self) -> str:
        """
        Scrapes the documentation from the provided URL.
        For now, uses the CatBoost documentation as an example.
        """
        try:
            response = requests.get(self.link_to_documentation, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract text content from the page
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading/trailing space
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit text length to avoid token limits (keep first 50000 chars)
            if len(text) > 50000:
                text = text[:50000] + "\n... [documentation truncated for length]"
            
            return text
            
        except requests.RequestException as e:
            print(f"Error scraping documentation: {e}")
            return f"Error: Could not retrieve documentation from {self.link_to_documentation}"
    
    def query_llm(self, scraped_data_documentation: str, baseline_data: Optional[Dict], 
                  past_proposals: List[Dict], impact_of_past_proposals: List[Dict], KB) -> Dict:
        """
        Queries the LLM to generate a KB change proposal.
        """
        prompt = f"""You are an expert in analyzing library documentation to extract Knowledge Base entries for the VAMSA data provenance tracking system.

        CRITICAL: KNOWLEDGE BASE ENTRY FORMAT
        Each KB entry is an annotation that VAMSA uses to track data transformations. The format is:

        **Annotation Structure:**
        - library: str (e.g., "pandas", "sklearn", "numpy", "catboost")
        - api_name: str (e.g., "fillna", "fit_transform", "merge", "CatBoostClassifier.fit")
        - inputs: List[str] - Types of data consumed (e.g., ["features"], ["data", "labels"], ["model"])
        - outputs: List[str] - Types of data produced (e.g., ["features"], ["model"], ["predictions"])
        - caller: str - One of: "data", "model", "metric"
        * "data" = transforms/manipulates datasets
        * "model" = creates or trains ML models
        * "metric" = evaluates/measures performance
        - transformation_type: str (optional) - Semantic description (e.g., "imputation", "scaling", "encoding", "training")

        **Example KB Entries (already in KB) -- so do not replicate:**
        ```json
                {{
                    "Library": "catboost",
                    "Module": None,
                    "Caller": "model",
                    "API Name": "fit",
                    "Inputs": ["features", "labels"],
                    "Outputs": ["trained model"],
                }},
                {{
                    "Library": "sklearn",
                    "Module": "model.selection",
                    "Caller": None,
                    "API Name": "train_test_split",
                    "Inputs": ["features", "labels"],
                    "Outputs": [
                        "features",
                        "validation features",
                        "labels",
                        "validation labels",
                    ],
                }},
                {{
                    "Library": "pandas",
                    "Module": None,
                    "Caller": None,
                    "API Name": "read_csv",
                    "Inputs": ["file_path"],
                    "Outputs": ["data"],
                }},
                {{
                    "Library": "pandas",
                    "Module": None,
                    "Caller": None,
                    "API Name": "concat",
                    "Inputs": ["data"],
                    "Outputs": ["data"],
                }},
        ```

        INPUT DATA:

        LIBRARY DOCUMENTATION (scraped from {self.link_to_documentation}):
        {scraped_data_documentation}

        BASELINE METRICS (current KB coverage):
        {json.dumps(baseline_data, indent=2) if baseline_data else "No baseline data available"}

        CURRENT KB ENTRIES:
        {json.dumps(KB.knowledge_base.to_dict(orient="records"), indent=2) if past_proposals else "No entries available"}

        IMPACT OF PAST PROPOSALS:
        {json.dumps(impact_of_past_proposals, indent=2) if impact_of_past_proposals else "No impact data available"}

        YOUR TASK:
        1. Analyze the library documentation to identify important APIs used in data science workflows
        2. Focus on APIs that:
        - Transform data (cleaning, preprocessing, feature engineering)
        - Create or train models
        - Evaluate models or compute metrics
        - Are commonly used but missing from baseline metrics
        3. Extract the API signature to determine correct inputs/outputs
        4. Classify the caller type based on what the API does
        5. Avoid duplicating past proposals
        6. Consider past proposal impacts to prioritize high-value additions

        Provide ONE high-priority KB entry that will most improve VAMSA's ability to track data transformations in this library."""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert in analyzing library documentation and extracting structured KB entries for data provenance systems."
                },
                {"role": "user", "content": prompt}
            ]
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_model=ProposalResponse,
            )
            
            # Convert Pydantic model to dict
            proposal_dict = {
                "description": response.description,
                "rationale": response.rationale,
                "expected_impact": response.expected_impact,
                "change_type": response.change_type,
                "details": {
                    "library": response.details.library,
                    "api_name": response.details.api_name,
                    "inputs": response.details.inputs,
                    "outputs": response.details.outputs,
                    "caller": response.details.caller,
                    "module": response.details.module,
                    "transformation_type": response.details.transformation_type
                }
            }
            
            return proposal_dict
            
        except Exception as e:
            print(f"Error querying LLM: {e}")
            # Return a fallback proposal
            return {
                "description": "Error generating proposal",
                "rationale": f"LLM query failed: {str(e)}",
                "expected_impact": "Unknown",
                "change_type": "annotation_entry",
                "details": {
                    "library": "unknown",
                    "api_name": "unknown",
                    "inputs": [],
                    "outputs": [],
                    "caller": "data",
                    "transformation_type": None
                }
            }

    def __call__(self, KB, baseline_report) -> 'KBChangeProposal':
        """
        Main execution method that orchestrates the proposal generation process.
        """
        # Scrape documentation
        print(f"Scraping documentation from {self.link_to_documentation}...")
        scraped_data = self.scrape_documentation()
        
        # Get past data on proposals
        past_proposals = self.past_proposals

        # Query LLM
        print("Querying LLM for proposal...")
        proposal_dict = self.query_llm(
            scraped_data,
            baseline_report,
            past_proposals,
            self.impact_of_past_proposals,
            KB
        )
        
        # Create KBChangeProposal object
        
        proposal = KBChangeProposal(
            library=proposal_dict["details"].get("library", {}),
            module=proposal_dict["details"].get("module", None),
            caller=proposal_dict["details"].get("caller", None),
            api_name=proposal_dict["details"].get("api_name", None),
            inputs=proposal_dict["details"].get("inputs", []),
            outputs=proposal_dict["details"].get("outputs", []),
            change_type=proposal_dict.get("change_type", "annotation_entry"),
            description=proposal_dict.get("description", ""),
            rationale=proposal_dict.get("rationale", ""),
        )
        
        # Store this proposal in history
        self.past_proposals.append(proposal_dict)
        
        return proposal


# Example usage:
if __name__ == "__main__":
    maker = ProposalMaker("https://catboost.ai/docs/en/concepts/python-reference_catboost")
    proposal = maker()
    print("\n=== Generated Proposal ===")
    print(f"Description: {proposal.description}")
    print(f"Rationale: {proposal.rationale}")
    print(f"Expected Impact: {proposal.expected_impact}")
    print(f"Details: {json.dumps(proposal.details, indent=2)}")