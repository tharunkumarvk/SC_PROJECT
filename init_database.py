import os
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'disease_prediction.settings')
django.setup()

from fuzzy_app.models import Symptom, Disease, DiseaseRule

# Clear existing data
DiseaseRule.objects.all().delete()
Disease.objects.all().delete()
Symptom.objects.all().delete()

# Create symptoms
symptoms_data = [
    {'name': 'Fever', 'min_value': 95, 'max_value': 105, 'unit': '°F'},
    {'name': 'Headache', 'min_value': 0, 'max_value': 10, 'unit': 'scale'},
    {'name': 'Cough', 'min_value': 0, 'max_value': 10, 'unit': 'scale'},
    {'name': 'Fatigue', 'min_value': 0, 'max_value': 10, 'unit': 'scale'},
    {'name': 'Body Pain', 'min_value': 0, 'max_value': 10, 'unit': 'scale'},
]

for symptom_data in symptoms_data:
    Symptom.objects.create(**symptom_data)

# Create diseases
diseases_data = [
    {'name': 'Common Cold', 'description': 'Viral infection of the upper respiratory tract'},
    {'name': 'Flu', 'description': 'Influenza viral infection affecting the respiratory system'},
    {'name': 'Malaria', 'description': 'Mosquito-borne infectious disease causing fever and chills'},
    {'name': 'Dengue', 'description': 'Mosquito-borne tropical disease causing high fever'},
]

disease_objs = {}
for disease_data in diseases_data:
    disease = Disease.objects.create(**disease_data)
    disease_objs[disease_data['name']] = disease

# Create rules
rules_data = [
    {'disease': 'Common Cold', 'symptom': 'Fever', 'severity': 'low', 'weight': 0.7},
    {'disease': 'Common Cold', 'symptom': 'Headache', 'severity': 'low', 'weight': 0.6},
    {'disease': 'Common Cold', 'symptom': 'Cough', 'severity': 'medium', 'weight': 0.8},
    {'disease': 'Flu', 'symptom': 'Fever', 'severity': 'medium', 'weight': 0.9},
    {'disease': 'Flu', 'symptom': 'Headache', 'severity': 'high', 'weight': 0.8},
    {'disease': 'Flu', 'symptom': 'Body Pain', 'severity': 'high', 'weight': 0.9},
    {'disease': 'Malaria', 'symptom': 'Fever', 'severity': 'high', 'weight': 1.0},
    {'disease': 'Malaria', 'symptom': 'Headache', 'severity': 'medium', 'weight': 0.7},
    {'disease': 'Malaria', 'symptom': 'Fatigue', 'severity': 'high', 'weight': 0.8},
    {'disease': 'Dengue', 'symptom': 'Fever', 'severity': 'high', 'weight': 1.0},
    {'disease': 'Dengue', 'symptom': 'Headache', 'severity': 'high', 'weight': 0.9},
    {'disease': 'Dengue', 'symptom': 'Body Pain', 'severity': 'high', 'weight': 0.9},
]

for rule_data in rules_data:
    DiseaseRule.objects.create(
        disease=disease_objs[rule_data['disease']],
        symptom=Symptom.objects.get(name=rule_data['symptom']),
        severity=rule_data['severity'],
        weight=rule_data['weight']
    )

print('Database initialized successfully!')
print(f'Symptoms: {Symptom.objects.count()}')
print(f'Diseases: {Disease.objects.count()}')
print(f'Rules: {DiseaseRule.objects.count()}')
