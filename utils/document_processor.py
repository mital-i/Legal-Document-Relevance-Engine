import datasets
from bs4 import BeautifulSoup
import re
import pandas as pd

# Load datasets
# legalbench = datasets.load_dataset("nguha/legalbench") # This line causes the error
legalbench = datasets.load_dataset("nguha/legalbench", "abercrombie") # Fixed line
try:
    caselaw = datasets.load_dataset("HFforLegal/case-law", streaming=True)
    print(caselaw)
except Exception as e:
    print(f"Error loading dataset: {e}")

def clean_document(text):
    # Remove HTML tags if present
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\(\)\[\]\{\}\-\"\'\?]', '', text)
    
    # Standardize line breaks
    text = re.sub(r'(\r\n|\r|\n)', '\n', text)
    
    return text.strip()

def segment_document(text):
    # More robust pattern for legal document sections
    section_headers = re.finditer(r'(ARTICLE\s+[IVXLCDM]+\.?\s*[A-Z][A-Za-z\s]+|[0-9]+\.[0-9]+\s+[A-Z][A-Za-z\s]+)', text)
    
    sections = []
    start_positions = []
    headers = []
    
    for match in section_headers:
        start_positions.append(match.start())
        headers.append(match.group().strip())
    
    # Add the end of the document
    start_positions.append(len(text))
    
    # Create sections based on the header positions
    for i in range(len(headers)):
        start = start_positions[i]
        end = start_positions[i+1] if i+1 < len(start_positions) else len(text)
        
        section_content = text[start:end].strip()
        sections.append({
            "title": headers[i],
            "content": section_content
        })
    
    # If no sections found, create one for the entire document
    if not sections:
        sections = [{"title": "Entire Document", "content": text}]
    
    return sections

def annotate_document(sections):
    annotations = []
    
    # Define keyword patterns for different clause types
    patterns = {
        "obligation": [r'shall', r'must', r'required to', r'obligated', r'duty to'],
        "right": [r'entitled to', r'right to', r'may', r'permitted to', r'option to'],
        "liability": [r'liable', r'damages', r'indemnify', r'warranty', r'disclaimer'],
        "termination": [r'terminate', r'cancellation', r'expiration', r'end of term'],
        "payment": [r'payment', r'fee', r'cost', r'expense', r'price', r'compensation']
    }
    
    for section in sections:
        section_annotations = {"section": section["title"], "labels": []}
        
        for label, keywords in patterns.items():
            pattern = '|'.join(keywords)
            if re.search(pattern, section["content"], re.IGNORECASE):
                section_annotations["labels"].append(label)
        
        annotations.append(section_annotations)
    
    return annotations

def process_legal_document(doc_text):
    # Clean
    cleaned_text = clean_document(doc_text)
    
    # Segment
    sections = segment_document(cleaned_text)
    
    # Annotate (choose one method)
    annotations = annotate_document(sections)
    # Or model-based: annotations = model_based_annotation(sections)
    
    return {
        "clean_text": cleaned_text,
        "sections": sections,
        "annotations": annotations
    }

# Process a sample document
sample_doc = legalbench["train"][1]["text"]  # Adjust according to actual dataset structure
result = process_legal_document(sample_doc)

print(result)