import spacy
from transformers import pipeline
import re

class ObligationRightsExtractor:
    def __init__(self):
        # Load spaCy model for dependency parsing
        self.nlp = spacy.load("en_core_web_sm")
        
        # Load zero-shot classifier for ambiguous cases
        self.zero_shot = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Obligation indicators 
        self.obligation_patterns = [
            r'\b(?:shall|must|required to|obligated to|has a duty to|is obliged to|will|agrees to)\b',
            r'\b(?:responsible for|liable for|bound to|committed to)\b',
            r'\b(?:is required|are required|be required)\b'
        ]
        
        # Rights indicators
        self.rights_patterns = [
            r'\b(?:may|can|is entitled to|has the right to|is authorized to)\b',
            r'\b(?:is permitted to|are permitted to|has the option to|reserves the right)\b',
            r'\b(?:is allowed to|are allowed to|has liberty to|has freedom to)\b'
        ]
        
        # Party identifiers
        self.party_patterns = [
            r'\b(?:Buyer|Seller|Lessor|Lessee|Licensor|Licensee|Contractor|Client)\b',
            r'\b(?:Employer|Employee|Landlord|Tenant|Vendor|Customer|Provider|Recipient)\b',
            r'\b(?:Company|User|Subscriber|Member|Patient|Insurer|Insured|Owner)\b'
        ]
        
    def extract_sentences(self, text):
        """Split text into sentences"""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]
    
    def classify_sentence(self, sentence):
        """Classify a sentence as obligation, right, or other"""
        # Check for obligation patterns
        for pattern in self.obligation_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return "obligation"
        
        # Check for rights patterns
        for pattern in self.rights_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return "right"
        
        # Use zero-shot for ambiguous cases
        result = self.zero_shot(
            sentence, 
            candidate_labels=["obligation", "right", "neither"],
            multi_label=False
        )
        
        # Only return obligation or right if confidence is high enough
        if result["scores"][0] > 0.7 and result["labels"][0] != "neither":
            return result["labels"][0]
        
        return "other"
    
    def identify_party(self, sentence):
        """Identify which party has the obligation or right"""
        doc = self.nlp(sentence)
        party = None
        
        # First check for explicit party mentions
        for pattern in self.party_patterns:
            match = re.search(pattern, sentence, re.IGNORECASE)
            if match:
                party = match.group()
                break
        
        # If no explicit party, try to find the subject of modal verbs
        if not party:
            for token in doc:
                if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                    # Check if the verb is a modal or in our patterns
                    if token.head.pos_ == "AUX" or any(re.search(p, token.head.text, re.IGNORECASE) 
                                                     for p in self.obligation_patterns + self.rights_patterns):
                        party = token.text
                        # Get the full noun phrase
                        for child in token.children:
                            if child.dep_ in ("compound", "amod", "det"):
                                party = f"{child.text} {party}"
                        break
        
        return party if party else "Unspecified Party"
    
    def extract_action(self, sentence):
        """Extract the action (what needs to be done)"""
        doc = self.nlp(sentence)
        
        # Find the main verb and its object
        main_verb = None
        verb_obj = None
        
        for token in doc:
            # Find modal + verb or obligation/right verb
            if (token.pos_ == "AUX" and token.dep_ == "ROOT") or \
               (token.pos_ == "VERB" and token.dep_ == "ROOT"):
                # Get the main verb
                if token.pos_ == "AUX":
                    # Look for the next verb
                    for child in token.children:
                        if child.pos_ == "VERB":
                            main_verb = child
                            break
                else:
                    main_verb = token
                
                # If we found a main verb, extract its object
                if main_verb:
                    verb_phrase = main_verb.text
                    
                    # Get any adverbial modifiers
                    adverbs = [child.text for child in main_verb.children if child.dep_ == "advmod"]
                    if adverbs:
                        verb_phrase = f"{' '.join(adverbs)} {verb_phrase}"
                    
                    # Get the object
                    obj_tokens = []
                    for child in main_verb.children:
                        if child.dep_ in ("dobj", "pobj", "attr"):
                            # Get the full noun phrase
                            obj_phrase = self._get_phrase(child)
                            obj_tokens.append(obj_phrase)
                    
                    if obj_tokens:
                        verb_obj = f"{verb_phrase} {' '.join(obj_tokens)}"
                    else:
                        verb_obj = verb_phrase
                    
                    break
        
        # If parsing failed, fall back to regex
        if not verb_obj:
            # Try to get the action after modal verbs or obligation indicators
            combined_patterns = self.obligation_patterns + self.rights_patterns
            for pattern in combined_patterns:
                match = re.search(f"{pattern}\\s+(.*?)(?:\\.|,|;|:)", sentence, re.IGNORECASE)
                if match:
                    verb_obj = match.group(1).strip()
                    break
        
        return verb_obj if verb_obj else "Unspecified Action"
    
    def _get_phrase(self, token):
        """Helper to get a full phrase from a root token"""
        phrase = token.text
        prefix = ""
        
        # Get all the children that come before
        pre_children = sorted([child for child in token.children 
                              if child.dep_ in ("amod", "compound", "det", "nummod") 
                              and child.i < token.i], key=lambda x: x.i)
        
        for child in pre_children:
            prefix += child.text + " "
        
        # Get all the children that come after
        post_children = sorted([child for child in token.children 
                               if child.dep_ in ("prep") 
                               and child.i > token.i], key=lambda x: x.i)
        
        suffix = ""
        for child in post_children:
            suffix += " " + child.text
            # Get the object of the preposition
            for grandchild in child.children:
                if grandchild.dep_ == "pobj":
                    suffix += " " + self._get_phrase(grandchild)
        
        return prefix + phrase + suffix
    
    def extract_conditions(self, sentence):
        """Extract conditions under which the obligation or right applies"""
        # Look for conditional indicators
        condition_indicators = [
            "if", "when", "provided that", "so long as", "in the event", 
            "subject to", "unless", "except if", "on condition that", "in case"
        ]
        
        conditions = []
        doc = self.nlp(sentence)
        
        # Look for adverbial clauses
        for token in doc:
            if token.dep_ == "mark" and token.text.lower() in condition_indicators:
                # Get the clause
                clause_tokens = [token.text]
                head = token.head
                
                # Get all descendants of the clause head
                clause_span = doc[head.left_edge.i: head.right_edge.i + 1]
                conditions.append(clause_span.text)
        
        # If parsing fails, try regex
        if not conditions:
            for indicator in condition_indicators:
                pattern = f"(?:{indicator})\\s+(.*?)(?=\\.|,|;|$)"
                matches = re.finditer(pattern, sentence, re.IGNORECASE)
                conditions.extend([match.group(0) for match in matches])
        
        return conditions if conditions else None
    
    def extract_from_text(self, text):
        """Extract obligations and rights from the full text"""
        sentences = self.extract_sentences(text)
        results = {
            "obligations": [],
            "rights": [],
            "other": []
        }
        
        for sentence in sentences:
            sentence_type = self.classify_sentence(sentence)
            
            if sentence_type in ["obligation", "right"]:
                party = self.identify_party(sentence)
                action = self.extract_action(sentence)
                conditions = self.extract_conditions(sentence)
                
                item = {
                    "sentence": sentence,
                    "party": party,
                    "action": action,
                    "conditions": conditions
                }
                
                results[sentence_type + "s"].append(item)
            else:
                results["other"].append({"sentence": sentence})
        
        return results
    
    def extract_from_sections(self, sections):
        """Extract obligations and rights from document sections"""
        results = []
        
        for section in sections:
            section_title = section["title"]
            section_text = section["content"]
            
            extractions = self.extract_from_text(section_text)
            
            results.append({
                "section_title": section_title,
                "extractions": extractions
            })
        
        return results

# Usage
extractor = ObligationRightsExtractor()
# results = extractor.extract_from_text(legal_text)
# section_results = extractor.extract_from_sections(document_sections)