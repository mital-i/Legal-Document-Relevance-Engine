from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import re

def implement_legal_ner(text):
    # Option 1: Use a specialized legal NER model
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")
    model = AutoModelForTokenClassification.from_pretrained("nlpaueb/legal-bert-small-uncased")
    
    # Create NER pipeline
    ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
    
    # Option 2: Rule-based NER for specialized legal entities
    # These patterns complement model-based NER for legal-specific entities
    patterns = {
        "DATE": r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2}',
        "AMOUNT": r'\$\s*\d+(?:,\d{3})*(?:\.\d{2})?|\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD)',
        "PARTY": r'(?:Plaintiff|Defendant|Petitioner|Respondent|Claimant)(?:\s[A-Z][a-z]+)+|(?:[A-Z][a-z]*\.?\s)*(?:[A-Z][a-z]+)(?:\s(?:LLC|Inc\.|Corporation|Corp\.|Ltd\.|Limited|Company|Co\.))',
        "COURT": r'(?:Supreme|District|Circuit|Federal|State|County|Municipal)\s+Court(?:\s+of\s+[A-Z][a-z]+)*',
        "REGULATION": r'(?:Section|§)\s+\d+(?:\.\d+)*(?:\([a-z]\))*\s+of\s+(?:the\s+)?(?:[A-Z][a-z]+\s+)+(?:Act|Code|Statute|Regulation)'
    }
    
    # Process document in chunks due to model token limitations
    chunks = [text[i:i+512] for i in range(0, len(text), 400)]  # Overlap by 112 tokens
    
    # Model-based NER
    all_entities = []
    for chunk in chunks:
        entities = ner(chunk)
        for entity in entities:
            # Adjust character spans to account for chunking
            chunk_offset = chunks.index(chunk) * 400
            entity['start'] += chunk_offset
            entity['end'] += chunk_offset
            all_entities.append(entity)
    
    # Rule-based NER to supplement
    for entity_type, pattern in patterns.items():
        matches = re.finditer(pattern, text)
        for match in matches:
            all_entities.append({
                'entity_group': entity_type,
                'score': 1.0,  # Rule-based confidence
                'word': match.group(),
                'start': match.start(),
                'end': match.end()
            })
    
    # Merge overlapping entities, prioritizing those with higher scores
    all_entities.sort(key=lambda x: (x['start'], -x['score']))
    merged_entities = []
    
    for entity in all_entities:
        if not merged_entities or entity['start'] >= merged_entities[-1]['end']:
            merged_entities.append(entity)
        elif entity['score'] > merged_entities[-1]['score']:
            # Replace with higher confidence entity
            merged_entities[-1] = entity
    
    # Group entities by type
    entity_groups = {}
    for entity in merged_entities:
        entity_type = entity['entity_group']
        if entity_type not in entity_groups:
            entity_groups[entity_type] = []
        entity_groups[entity_type].append(entity['word'])
    
    return {
        'detailed_entities': merged_entities,
        'grouped_entities': entity_groups
    }