# HIV Clinical Nudge Engine - Project Brief
## MedGemma Impact Challenge Submission

---

## 1. High-Level Goal

Build a Retrieval-Augmented Generation (RAG) based Clinical Decision Support System that analyzes patient records against Uganda Consolidated HIV Guidelines and generates actionable "nudges" to help clinicians catch missed monitoring, identify drug toxicities, and optimize care delivery at the point of care. The system will be demonstrated as a Streamlit-based proof-of-concept for the MedGemma Impact Challenge.

---

## 2. Core Features & Requirements

### Core Functionality
- **Guideline Processing**: Ingest Uganda Consolidated HIV Guidelines (PDF), chunk into semantic segments, and create vector embeddings for retrieval
- **Patient Data Analysis**: Process both structured data (labs, medications, visit dates) and unstructured clinical notes (free-text, potentially multilingual English/Luganda)
- **Clinical Entity Extraction**: Extract symptoms, drug names, temporal information, and psychosocial factors from narrative notes using MedGemma or similar NLP models
- **Semantic Retrieval**: Match patient clinical context against guideline database using vector similarity search
- **Nudge Generation**: Generate prioritized, actionable clinical recommendations with three levels:
  - **HIGH**: Safety-critical issues (overdue monitoring, potential toxicities)
  - **MEDIUM**: Optimization opportunities (DSD eligibility, regimen simplification)
  - **LOW**: Informational items (general guidance)
- **Guideline Traceability**: Each nudge must link to specific guideline section that triggered it

### Demo-Specific Requirements
- **Dual-Mode Operation**:
  - Real-time mode: Lightweight keyword-based alerts during documentation
  - Post-visit mode: Comprehensive RAG-based analysis after visit completion
- **Interactive Patient Selection**: Demo with 3-5 realistic mock patients covering different scenarios
- **Visual Pipeline**: Show the RAG workflow steps (Extract → Retrieve → Generate)
- **Expandable Details**: Allow judges to inspect retrieved guideline chunks and extracted entities

### Clinical Scenarios to Demonstrate
1. **Missed Monitoring**: Patient on TDF with creatinine >12 months overdue
2. **Drug Toxicity Detection**: Patient on EFV with multiple mentions of sleep disturbance
3. **DSD Eligibility**: Stable patient (VL suppressed >2 years) eligible for less frequent visits
4. **Multilingual Note Processing**: Clinical note mixing English and local language terms (demonstrate capability)

---

## 3. Technology Stack

### Programming Language
- **Primary**: Python 3.9+

### Core Frameworks & Libraries
- **Streamlit**: Web UI for interactive demo (v1.28+)
- **LangChain**: RAG orchestration framework
- **Sentence Transformers**: Text embeddings (`all-MiniLM-L6-v2` for demo, upgradeable to BiomedBERT)
- **Vector Database**: 
  - Pinecone (cloud option for easy deployment) OR
  - ChromaDB (local/open-source option for cost-effective demo)
- **MedGemma** (target) or fallback to GPT-3.5-turbo for text generation

### Data Processing
- **PyPDF2** or **pdfplumber**: Extract text from PDF guidelines
- **spaCy** or **Hugging Face Transformers**: Clinical entity extraction
- **Pandas**: Structured data manipulation
- **NumPy**: Vector operations and similarity calculations

### Optional/Enhancement
- **mBART** or similar: For Luganda-English translation (leverage existing work)
- **Plotly**: For visualization of retrieval scores/confidence
- **dotenv**: Environment variable management for API keys

---

## 4. Code Examples

Reference the modular architecture discussed:

### Project Structure
```
streamlit-hiv-nudge-demo/
├── app.py                          # Main Streamlit application
├── data/
│   ├── uganda_guidelines.pdf       # Source guidelines
│   ├── guidelines_chunks.json      # Pre-processed embeddings
│   └── mock_patients.json          # Demo patient data
├── modules/
│   ├── guideline_processor.py      # PDF → chunks → embeddings
│   ├── patient_parser.py           # Extract clinical entities
│   ├── rag_engine.py               # Retrieval & context matching
│   └── nudge_generator.py          # Business logic for nudges
├── utils/
│   ├── embeddings.py               # Embedding utilities
│   └── mock_data.py                # Generate realistic test data
├── requirements.txt
└── README.md
```

### Key Code Pattern: RAG Retrieval
```python
# Example from rag_engine.py
def retrieve_guidelines(self, clinical_context, top_k=5):
    """Semantic search for relevant guidelines"""
    # Create contextualized query
    query = f"""
    Patient on {clinical_context['regimen']} 
    for {clinical_context['duration_on_art']} months.
    Symptoms: {clinical_context['clinical_mentions']}
    """
    
    # Embed and retrieve
    query_embedding = self.embedder.encode(query)
    similarities = self.compute_similarity(query_embedding)
    return similarities[:top_k]
```

### Key Code Pattern: Nudge Generation
```python
# Example from nudge_generator.py
def _check_lab_monitoring(self, patient_data, context):
    """Rule-based + guideline-backed nudges"""
    nudges = []
    
    if 'TDF' in patient_data['art_regimen']:
        last_creatinine_date = patient_data['labs']['Creatinine']['date']
        months_overdue = calculate_months_since(last_creatinine_date)
        
        if months_overdue > 6:
            nudges.append({
                'priority': 'HIGH',
                'title': 'TDF Monitoring Overdue',
                'message': f'Creatinine last checked {months_overdue} months ago...',
                'guideline_ref': self.match_guideline('TDF monitoring')
            })
    
    return nudges
```

---

## 5. Documentation & References

### Official Guidelines
- **Uganda Consolidated HIV Guidelines** (2023 version): [Ministry of Health Uganda](https://www.health.go.ug/)
- Available at: https://elearning.idi.co.ug/pluginfile.php/11665/mod_resource/content/1/CONSOLIDATED-GUIDELINES-2023-Final.pdf

### Technical Documentation
- **MedGemma Models**: [Google Health AI Developer Foundations](https://developers.google.com/health-ai)
- **LangChain RAG**: https://python.langchain.com/docs/use_cases/question_answering/
- **Sentence Transformers**: https://www.sbert.net/
- **Streamlit**: https://docs.streamlit.io/
- **Pinecone Vector DB**: https://docs.pinecone.io/
- **ChromaDB** (open-source alternative): https://docs.trychroma.com/

### Research References
- **RAG in Healthcare Survey**: [MDPI - Comprehensive Review](https://www.mdpi.com/2673-2688/6/9/226)
- **Clinical Decision Support Meta-Analysis**: [JAMIA - Systematic Review](https://academic.oup.com/jamia/article/32/4/605/7954485)
- **Uganda HIV Program**: [PEPFAR Uganda](https://www.state.gov/pepfar-uganda/)

### Related Clinical Tools
- **Apollo 24|7 CIE**: Real-world RAG implementation with MedPaLM
- **Surgical Fitness Assessment System**: RAG with 35+ guidelines achieving 96.4% accuracy

---

## 6. Other Considerations & Gotchas

### Data Privacy & Security
- **CRITICAL**: Demo uses synthetic data only - no real patient information
- If deploying: Must ensure HIPAA/Uganda Data Protection Act compliance
- Vector database must be secured (encrypted at rest/transit)
- Audit logging for all nudge generations

### Multilingual Processing Challenges
- Clinical notes in Uganda often mix English with Luganda, Runyankole, etc.
- **Strategy**: 
  1. Use language detection to identify mixed-language segments
  2. Translate non-English portions using mBART or similar
  3. Run entity extraction on unified English text
- **Fallback**: For demo, include 1-2 examples with common Luganda medical terms to show awareness

### Guideline Versioning
- HIV guidelines update periodically (every 2-3 years)
- **Solution**: Version stamp guideline embeddings, allow hot-swapping of knowledge base
- Add metadata: `{chunk_id, guideline_version, last_updated}`

### Context Window Limitations
- Patient history can be extensive (years of notes)
- **Strategy**: 
  - Retrieve last 6 months of clinical notes by default
  - For chronic issues, expand window based on query
  - Summarize older history before embedding

### Computational Resources
- **Embedding Generation**: ~2-3 seconds for 1000 guideline chunks (one-time)
- **Query Time**: <500ms for similarity search + generation
- **Demo Hardware**: Can run on standard laptop (no GPU required for inference with sentence-transformers)
- **Production**: Consider GPU for MedGemma fine-tuning (A100 recommended)

### False Positives/Negatives
- **Risk**: System might over-alert or miss subtle issues
- **Mitigation**:
  - Implement confidence thresholds (only show nudges >0.7 similarity)
  - Allow clinician feedback loop (thumbs up/down on nudges)
  - Include "Why this nudge?" explanation with guideline excerpt

### Integration with EMR Systems
- Demo is standalone, but real deployment requires EMR integration
- **Considerations**:
  - HL7 FHIR standard for data exchange
  - Webhook triggers on visit completion
  - API endpoint for nudge retrieval
  - Display in existing EMR UI vs. separate dashboard

### Ethical Considerations
- System provides **decision support**, not **decision making**
- Must clearly state: "Clinician has final authority"
- Avoid language that sounds prescriptive ("You must...") → use suggestive ("Consider...")
- Include disclaimer in demo UI

---

## 7. Success Criteria

### Technical Success Metrics
- ✅ System successfully processes Uganda HIV Guidelines (300+ pages) into searchable vector database
- ✅ Accurately extracts clinical entities from at least 90% of test cases (drug names, symptoms, dates)
- ✅ Retrieval system returns relevant guideline sections with >0.75 similarity score for key scenarios
- ✅ Demo runs smoothly with <3 second response time for post-visit analysis
- ✅ Real-time keyword detection responds within 500ms

### Clinical Validity Metrics
- ✅ Generated nudges align with Uganda HIV Guidelines (verified by clinical reviewer)
- ✅ High-priority alerts correctly identify all safety-critical scenarios (overdue monitoring, known toxicities)
- ✅ Zero false high-priority alerts in test cases (no "crying wolf")
- ✅ DSD eligibility correctly identified for stable patients (VL suppressed >12mo, adherence good)

### Demo Presentation Metrics
- ✅ Clear visualization of RAG pipeline (judges can see retrieve → augment → generate)
- ✅ Guideline traceability: each nudge links to source text
- ✅ At least 3 diverse patient scenarios demonstrated
- ✅ Show both real-time and post-visit modes
- ✅ Interface is intuitive (judges can navigate without tutorial)

### Impact Potential Metrics
- ✅ Articulate clear value proposition: "Reduces missed monitoring by X%, identifies toxicity Y% faster"
- ✅ Demonstrate scalability: modular design allows adaptation to TB, maternal health, etc.
- ✅ Show EMR integration pathway (even if not implemented)
- ✅ Address multilingual capability (even if simplified in demo)

### Competition-Specific Criteria
- ✅ Leverages MedGemma capabilities for clinical entity extraction
- ✅ Addresses real healthcare challenge in resource-constrained setting
- ✅ System is auditable (can explain why each nudge was generated)
- ✅ Demonstrates potential for real-world deployment

---

## 8. ML Project Flag

**is_ml_project: true**

### Justification
This is an ML/AI project involving:
- Natural Language Processing for clinical entity extraction
- Vector embeddings and semantic search
- Large Language Model integration (MedGemma)
- RAG architecture combining retrieval and generation

### ML-Specific Considerations
- **Model Selection**: Compare MedGemma vs. GPT-3.5 vs. BiomedBERT for entity extraction
- **Embedding Model**: Evaluate sentence-transformers vs. domain-specific biomedical embeddings
- **Evaluation Metrics**: 
  - Precision/Recall for entity extraction
  - Retrieval accuracy (% of correct guideline sections retrieved)
  - Nudge relevance score (clinical expert ratings)
- **Fine-tuning Strategy**: If time permits, fine-tune MedGemma on Uganda-specific clinical notes
- **Baseline Comparison**: Compare RAG approach vs. rule-based clinical decision support

---

## Additional Notes for Implementation

### Phase 1: Setup (Week 1)
- Set up development environment
- Extract and process Uganda HIV Guidelines
- Generate embeddings and populate vector database
- Create mock patient dataset

### Phase 2: Core Development (Week 2-3)
- Build RAG engine with retrieval logic
- Implement nudge generation rules
- Develop Streamlit UI with dual-mode functionality
- Integrate all modules

### Phase 3: Testing & Refinement (Week 4)
- Clinical validation with healthcare advisor
- Test with diverse patient scenarios
- Optimize retrieval parameters (chunk size, overlap, top_k)
- Prepare demo script and video

### Backup Plans
- **If MedGemma access delayed**: Use GPT-3.5-turbo with clinical prompting
- **If computational resources limited**: Use lighter embedding model (all-MiniLM-L6-v2)
- **If guideline PDF unavailable**: Use WHO HIV treatment guidelines as fallback
- **If vector DB complex**: Use in-memory numpy-based similarity search

---

## Contact & Team Information
- **Primary Developer**: [Your Name]
- **Clinical Advisor**: [If applicable]
- **Institution**: [Your Institution]
- **Competition**: MedGemma Impact Challenge
- **Submission Date**: [Target Date]

---

**Last Updated**: January 2025
**Version**: 1.0