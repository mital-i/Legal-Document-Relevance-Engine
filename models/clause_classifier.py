from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class ClauseClassifier:
    def __init__(self, model_name="nlpaueb/legal-bert-small-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=5  # Adjust based on your number of clause types
        )
        self.label_map = {
            0: "liability",
            1: "privacy",
            2: "payment",
            3: "termination",
            4: "rights"
        }
        
        # Fine-tune the model with your annotated data
        # self.fine_tune(training_data)
    
    def fine_tune(self, training_data):
        """
        Fine-tune the model with annotated clause data
        training_data: List of {"text": "clause text", "label": "clause_type"}
        """
        # Implementation omitted for brevity
        pass
        
    def classify_clause(self, clause_text):
        inputs = self.tokenizer(
            clause_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = F.softmax(outputs.logits, dim=-1)[0]
            
        results = {
            self.label_map[i]: float(score) 
            for i, score in enumerate(predictions)
        }
        
        # Get the highest scoring label
        predicted_label = max(results, key=results.get)
        confidence = results[predicted_label]
        
        return {
            "predicted_label": predicted_label,
            "confidence": confidence,
            "all_scores": results
        }
    
    def classify_document_sections(self, sections):
        """
        Classify multiple sections of a document
        sections: List of {"title": "section title", "content": "section text"}
        """
        results = []
        
        for section in sections:
            classification = self.classify_clause(section["content"])
            results.append({
                "section_title": section["title"],
                "section_text": section["content"][:100] + "...",  # Preview
                "classification": classification["predicted_label"],
                "confidence": classification["confidence"],
                "all_labels": classification["all_scores"]
            })
            
        return results

# Usage
clause_classifier = ClauseClassifier()