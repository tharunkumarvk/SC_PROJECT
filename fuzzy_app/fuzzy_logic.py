class FuzzySystem:
    def __init__(self):
        self.symptoms = {}
        self.diseases = {}
        self.rules = []
        self._load_from_db()

    def _load_from_db(self):
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
        for rule in DiseaseRule.objects.select_related('disease', 'symptom'):
            self.diseases[rule.disease.name]['rules'].append({
                'symptom': rule.symptom.name,
                'severity': rule.severity,
                'weight': rule.weight,
                'min': rule.threshold_min,
                'max': rule.threshold_max
            })

    def _trapezoidal_mf(self, x, a, b, c, d):
        """Safe trapezoidal membership function"""
        left = (x - a)/(b - a) if (b - a) != 0 else 1 if x >= b else 0
        right = (d - x)/(d - c) if (d - c) != 0 else 1 if x <= c else 0
        return max(min(left, 1, right), 0)

    def predict(self, symptom_values):
        results = {}
        
        # Calculate memberships
        memberships = {}
        for name, value in symptom_values.items():
            if name in self.symptoms:
                s = self.symptoms[name]
                normalized = (value - s['min'])/(s['max'] - s['min'])
                memberships[name] = {
                    'low': self._trapezoidal_mf(normalized, 0, 0, 0.3, 0.5),
                    'medium': self._trapezoidal_mf(normalized, 0.3, 0.5, 0.7),
                    'high': self._trapezoidal_mf(normalized, 0.5, 0.7, 1, 1)
                }
        
        # Evaluate rules
        for disease, data in self.diseases.items():
            confidence = 0
            for rule in data['rules']:
                if rule['symptom'] in memberships:
                    rule_confidence = memberships[rule['symptom']][rule['severity']] * rule['weight']
                    confidence = max(confidence, rule_confidence)
            
            if confidence > 0:
                results[disease] = {
                    'confidence': round(confidence, 2),
                    'description': data['description']
                }
        
        return sorted(results.items(), key=lambda x: x[1]['confidence'], reverse=True)