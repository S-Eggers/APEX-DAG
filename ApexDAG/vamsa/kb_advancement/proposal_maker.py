class ProposalMaker:
    def __init__(self, link_to_documentation: str):
        self.past_proposals = []
        self.link_to_documentation = link_to_documentation
        self.impact_of_past_proposals = []
    def scrape_documentation(self):
        # logic to scrape documentation
        pass
    
    def query_llm(self, scraped_data_documentation, baseline_data, past_proposals, impact_of_past_proposals):
        # logic to query llm
        prompt = f"""You are an expert in analyzing library documentation to extract Knowledge Base entries for the VAMSA data provenance tracking system.

        CRITICAL: KNOWLEDGE BASE ENTRY FORMAT
        Each KB entry is an annotation that VAMSA uses to track data transformations. The format is:

        **Annotation Structure:**
        - library: str (e.g., "pandas", "sklearn", "numpy")
        - api_name: str (e.g., "fillna", "fit_transform", "merge")
        - inputs: List[str] - Types of data consumed (e.g., ["features"], ["data", "labels"], ["model"])
        - outputs: List[str] - Types of data produced (e.g., ["features"], ["model"], ["predictions"])
        - caller: str - One of: "data", "model", "metric"
        * "data" = transforms/manipulates datasets
        * "model" = creates or trains ML models
        * "metric" = evaluates/measures performance
        - transformation_type: str (optional) - Semantic description (e.g., "imputation", "scaling", "encoding", "training")

        **Example KB Entries:**
        ```json
        {{
        "library": "pandas",
        "api_name": "fillna",
        "inputs": ["features"],
        "outputs": ["features"],
        "caller": "data",
        "transformation_type": "imputation"
        }}

        {{
        "library": "sklearn.preprocessing",
        "api_name": "StandardScaler.fit_transform",
        "inputs": ["features"],
        "outputs": ["features"],
        "caller": "data",
        "transformation_type": "scaling"
        }}

        {{
        "library": "sklearn.ensemble",
        "api_name": "RandomForestClassifier.fit",
        "inputs": ["features", "labels"],
        "outputs": ["model"],
        "caller": "model",
        "transformation_type": "training"
        }}
        ```

        INPUT DATA:

        LIBRARY DOCUMENTATION (scraped from {self.link_to_documentation}):
        {scraped_data_documentation}

        BASELINE METRICS (current KB coverage):
        {baseline_data}

        PAST PROPOSALS:
        {past_proposals}

        IMPACT OF PAST PROPOSALS:
        {impact_of_past_proposals}

        YOUR TASK:
        1. Analyze the library documentation to identify important APIs used in data science workflows
        2. Focus on APIs that:
        - Transform data (cleaning, preprocessing, feature engineering)
        - Create or train models
        - Are commonly used but missing from baseline metrics
        3. Extract the API signature to determine correct inputs/outputs
        4. Classify the caller type based on what the API does
        5. Avoid duplicating past proposals
        6. Consider past proposal impacts to prioritize high-value additions

        OUTPUT FORMAT (JSON):
        {{
            "description": "Brief description of what API this adds tracking for",
            "rationale": "Why this API is important for provenance tracking",
            "expected_impact": "Expected improvement in tracking coverage",
            "change_type": "annotation_entry",
            "details": {{
                "library": "exact_library_name",
                "api_name": "ClassName.method_name or function_name",
                "inputs": ["list", "of", "input_types"],
                "outputs": ["list", "of", "output_types"],
                "caller": "data|model|metric",
                "transformation_type": "semantic_description"
            }}
        }}

        Provide ONE high-priority KB entry that will most improve VAMSA's ability to track data transformations in this library."""
        
            
        

    def __call__(self) -> KBChangeProposal:
        # scrape documenttauon
        scraped_data = self.scrape_documentation()
        # get past data on proposals
        past_proposals = self.past_proposals
        # add report 
        baseline_data = None  # logic to get baseline data
        # query llm
        
        
        proposal = KBChangeProposal(
            description="Add fillna annotation for pandas",
            rationale="fillna is commonly used for handling missing data",
            expected_impact="Should improve coverage for data cleaning",
            change_type="annotation_entry",
            details={
                "library": "pandas",
                "api_name": "fillna",
                "inputs": ["features"],
                "outputs": ["features"],
                "caller": "data"
            }
        )
        return proposal