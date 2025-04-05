from models.ner import implement_legal_ner
from models.clause_classifier import ClauseClassifier
from models.extractor import ObligationRightsExtractor
from utils.document_processor import segment_document
import streamlit as st
class LegalDocumentAnalyzer:
    def __init__(self):
        self.ner = implement_legal_ner
        self.clause_classifier = ClauseClassifier()
        self.obligation_extractor = ObligationRightsExtractor()
    
    def analyze_document(self, document_text, user_profile=None):
        """
        Analyze a legal document and extract personalized insights
        
        Args:
            document_text: The text of the legal document
            user_profile: Optional dict with user concerns and preferences
                         e.g. {"concerns": ["privacy", "liability"], 
                               "role": "buyer"}
        """
        # 1. Run NER to identify key entities
        entities = self.ner(document_text)
        
        # 2. Segment the document into sections
        sections = segment_document(document_text)
        
        # 3. Classify each section by clause type
        classified_sections = self.clause_classifier.classify_document_sections(sections)
        
        # 4. Extract obligations and rights
        extraction_results = self.obligation_extractor.extract_from_sections(sections)
        
        # 5. Personalize results based on user profile
        if user_profile:
            personalized_results = self._personalize_results(
                classified_sections, 
                extraction_results, 
                entities, 
                user_profile
            )
        else:
            personalized_results = None
        
        return {
            "entities": entities,
            "sections": [
                {
                    "section_info": section_info,
                    "extractions": next((e for e in extraction_results 
                                         if e["section_title"] == section_info["section_title"]), None)
                }
                for section_info in classified_sections
            ],
            "personalized_insights": personalized_results
        }
    
    def _personalize_results(self, classified_sections, extraction_results, entities, user_profile):
        """Generate personalized insights based on user profile"""
        insights = []
        
        # Match user concerns with section classifications
        if "concerns" in user_profile:
            for concern in user_profile["concerns"]:
                matching_sections = [s for s in classified_sections 
                                    if s["classification"] == concern]
                
                for section in matching_sections:
                    insights.append({
                        "type": "concern_match",
                        "concern": concern,
                        "section": section["section_title"],
                        "importance": "high"
                    })
        
        # Match user role with obligations
        if "role" in user_profile:
            role = user_profile["role"].lower()
            
            for result in extraction_results:
                for obligation in result["extractions"]["obligations"]:
                    if role in obligation["party"].lower():
                        insights.append({
                            "type": "role_obligation",
                            "obligation": obligation["action"],
                            "section": result["section_title"],
                            "importance": "high"
                        })
        
        return insights

def format_results(results):
    """Format the analysis to display only summaries without raw legal text."""
    output = "=== LEGAL DOCUMENT ANALYSIS ===\n\n"
    
    # Format entities: print only counts and a few examples
    output += "KEY ENTITIES FOUND:\n"
    for entity_type, entity_list in results["entities"].get("grouped_entities", {}).items():
        if entity_list:
            count = len(entity_list)
            examples = entity_list[:-2]
            output += f"- {entity_type}: {count} found (e.g., {', '.join(examples)})\n"
    
    # Format sections: show only the title and classification, not the full text
    output += "\nDOCUMENT SECTIONS (Clause Classifications):\n"
    for section in results["sections"]:
        section_info = section["section_info"]
        title = section_info.get("section_title", "Unknown Section")
        classification = section_info.get("classification", "N/A")
        output += f"- {title} (Type: {classification})\n"
    
    # Format personalized insights (if any)
    if results.get("personalized_insights"):
        output += "\nPERSONALIZED INSIGHTS:\n"
        for insight in results["personalized_insights"]:
            if insight["type"] == "concern_match":
                output += f"- CONCERN: {insight['concern']} found in section '{insight['section']}'\n"
            elif insight["type"] == "role_obligation":
                obligation = insight.get("obligation", "take action")
                output += f"- OBLIGATION: As a party, you must {obligation} (in section '{insight['section']}')\n"
    else:
        output += "\nNo personalized insights available. Try providing a user profile with concerns and role.\n"
    
    return output

if __name__ == "__main__":
    # Sample document for testing
    uploaded_file = st.file_uploader("Upload a legal document to analyze")
    if uploaded_file is not None:
        # Read the file content directly from the UploadedFile object.
        file_bytes = uploaded_file.read()
        sample_document = file_bytes.decode("utf-8")
        # If your document is encoded in utf-8, decode it to a string.
    # uploaded_file = st.file_uploader("Uploaded a legal document to analyze")
    # with open(uploaded_file, "r") as f:
    #     sample_document = f.read()
    
    # Sample user profile
    user_profile = {
        "concerns": ["privacy", "liability"],
        "role": "buyer"
    }
    
    # Create analyzer and process document
    analyzer = LegalDocumentAnalyzer()
    results = analyzer.analyze_document(sample_document, user_profile)

    print(format_results(results))
    st.write(format_results(results))
    
    # Print some results
    # print("Document Entities:")
    # print(results["entities"]["grouped_entities"])
    # print("\nKey Sections by Type:")
    # for section in results["sections"]:
    #     print(f"- {section['section_info']['section_title']}: {section['section_info']['classification']}")
    
    # print("\nPersonalized Insights:")
    # for insight in results["personalized_insights"]:
    #     print(f"- {insight['type']}: {insight.get('obligation', insight.get('concern', ''))} in section {insight['section']}")
