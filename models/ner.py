import re

def implement_legal_ner(text):
    patterns = {
        "DATE": r'\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}|\d{4}-\d{2}-\d{2})\b',
        "AMOUNT": r'\b(\$\s*\d+(?:,\d{3})*(?:\.\d{2})?)\b|\b(\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars|USD))\b',
        "PARTY": r'\b((?:Client|Provider|Corporation|Company|Inc\.|LLC))\b',
        "PERSON": r"\b[A-Z][a-z]+(?:\s+(?:[A-Z]\.?|[A-Z][a-z]+))+\b",
        "LOCATION": r'\b([A-Z][a-z]+(?:town|city|ville|burg))\b',
        "TERM": r'\b((?:one|two|three|four|five|1|2|3|4|5)\s*(?:\(.*?\))?\s*(?:year|month|week|day))\b'
    }
    
    entity_groups = {}
    
    for entity_type, pattern in patterns.items():
        # Use case-insensitive matching for all except PERSON
        flags = re.IGNORECASE if entity_type != "PERSON" else 0
        matches = re.finditer(pattern, text, flags)
        found = set()
        for match in matches:
            # Some patterns (like AMOUNT) have multiple capturing groups; pick the first non-empty group.
            groups = match.groups()
            match_text = next((g for g in groups if g), match.group())
            match_text = match_text.strip()
            if match_text:
                found.add(match_text)
        entity_groups[entity_type] = list(found)
    
    return {'grouped_entities': entity_groups}
