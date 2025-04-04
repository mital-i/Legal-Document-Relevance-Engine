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
    # Split by common section headers in legal documents
    section_patterns = [
        r'(ARTICLE [A-Z0-9]+[\.:]|Section [0-9\.]+[\.:]|[0-9]+\.[0-9]+\.)',
        r'([A-Z][A-Z\s]+[\.:])',  # ALL CAPS HEADERS:
        r'(\n[IVX]+\.|\n[0-9]+\.)'  # Roman numerals or numbered lists
    ]
    
    sections = []
    last_pos = 0
    
    for pattern in section_patterns:
        matches = list(re.finditer(pattern, text))
        
        for i, match in enumerate(matches):
            start_pos = match.start()
            
            # Find the end position (start of next match or end of text)
            end_pos = matches[i+1].start() if i < len(matches)-1 else len(text)
            
            if start_pos > last_pos:
                # Add the section header and content
                section_title = match.group().strip()
                section_content = text[start_pos:end_pos].strip()
                sections.append({
                    "title": section_title,
                    "content": section_content
                })
                
            last_pos = end_pos
    
    # If no sections were found with the patterns, try splitting by paragraphs
    if not sections:
        paragraphs = text.split('\n\n')
        sections = [{"title": f"Paragraph {i+1}", "content": p.strip()} 
                   for i, p in enumerate(paragraphs) if p.strip()]
    
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