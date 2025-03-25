class FuzzySystem:
    def __init__(self):
        self.symptoms = {}
        self.diseases = {}
        self.rules = []

    def load_from_db(self):
        from .models import Symptom, Disease, DiseaseRule
        
        # Load symptoms
        for symptom in Symptom.objects.all():
            self.symptoms[symptom.name] = {
                'min': symptom.min_value,
                'max': symptom.max_value,
                'unit': symptom.unit
            }
        
        # Load diseases
        for disease in Disease.objects.all():
            self.diseases[disease.name] = {
                'description': disease.description,
                'rules': []
            }
        
        # Load rules
        for rule in DiseaseRule.objects.select_related('disease', 'symptom').all():
            self.diseases[rule.disease.name]['rules'].append({
                'symptom': rule.symptom.name,
                'severity': rule.severity,
                'weight': rule.weight
            })

    def triangular_mf(self, x, a, b, c):
        """Triangular membership function with edge case handling"""
        if b == a or c == b:
            return 0.0  # or handle differently based on your needs
            
        return max(min((x-a)/(b-a), (c-x)/(c-b)), 0)

    def trapezoidal_mf(self, x, a, b, c, d):
        """Trapezoidal membership function with edge case handling"""
        # Handle cases where denominators might be zero
        left = (x-a)/(b-a) if (b-a) != 0 else 1.0 if x >= b else 0.0
        right = (d-x)/(d-c) if (d-c) != 0 else 1.0 if x <= c else 0.0
        
        return max(min(left, 1, right), 0)

    def calculate_membership(self, symptom_name, value):
        """Calculate membership degrees for a symptom value with validation"""
        if symptom_name not in self.symptoms:
            return {}
        
        symptom = self.symptoms[symptom_name]
        min_val = symptom['min']
        max_val = symptom['max']
        
        # Validate input value
        if value < min_val or value > max_val:
            return {}
        
        # Define fuzzy sets for symptom with safe ranges
        low = self.trapezoidal_mf(value, 
                                 min_val, 
                                 min_val, 
                                 min_val + 0.3*(max_val-min_val), 
                                 min_val + 0.5*(max_val-min_val))
        
        medium = self.triangular_mf(value,
                                  min_val + 0.3*(max_val-min_val),
                                  min_val + 0.5*(max_val-min_val),
                                  min_val + 0.7*(max_val-min_val))
        
        high = self.trapezoidal_mf(value,
                                 min_val + 0.5*(max_val-min_val),
                                 min_val + 0.7*(max_val-min_val),
                                 max_val,
                                 max_val)
        
        return {
            'low': low,
            'medium': medium,
            'high': high
        }

    def predict_disease(self, symptom_values):
        """Predict diseases based on symptom values with input validation"""
        results = {}
        
        # Validate input
        if not symptom_values:
            return results
        
        # Calculate membership degrees for all symptoms
        memberships = {}
        for symptom_name, value in symptom_values.items():
            if symptom_name in self.symptoms:
                memberships[symptom_name] = self.calculate_membership(symptom_name, float(value))
        
        # Evaluate rules for each disease
        for disease_name, disease_data in self.diseases.items():
            max_confidence = 0
            
            for rule in disease_data['rules']:
                symptom_name = rule['symptom']
                severity = rule['severity']
                weight = rule['weight']
                
                if symptom_name in memberships and severity in memberships[symptom_name]:
                    rule_confidence = memberships[symptom_name][severity] * weight
                    max_confidence = max(max_confidence, rule_confidence)
            
            if max_confidence > 0:
                results[disease_name] = {
                    'confidence': round(max_confidence, 2),
                    'description': disease_data['description']
                }
        
        # Sort results by confidence
        return dict(sorted(results.items(), key=lambda item: item[1]['confidence'], reverse=True))