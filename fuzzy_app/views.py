from django.shortcuts import render, redirect
from .forms import SymptomInputForm
from .fuzzy_logic import FuzzySystem
from .models import Symptom, Disease, DiseaseRule  # Added DiseaseRule import

def index(request):
    if request.method == 'POST':
        form = SymptomInputForm(request.POST)
        if form.is_valid():
            # Prepare symptom values
            symptom_values = {}
            for field_name, value in form.cleaned_data.items():
                if value is not None and field_name.startswith('symptom_'):
                    symptom_id = int(field_name.split('_')[1])
                    symptom = Symptom.objects.get(id=symptom_id)
                    symptom_values[symptom.name] = value
            
            # Get prediction
            fuzzy_system = FuzzySystem()
            fuzzy_system.load_from_db()
            results = fuzzy_system.predict_disease(symptom_values)
            
            return render(request, 'fuzzy_app/results.html', {
                'results': results,
                'symptom_values': symptom_values
            })
    else:
        form = SymptomInputForm()
    
    return render(request, 'fuzzy_app/index.html', {'form': form})

def about(request):
    return render(request, 'fuzzy_app/about.html')

def init_db(request):
    """Initialize database with sample data (for development)"""
    if not Symptom.objects.exists():
        # Create symptoms
        symptoms = [
            {'name': 'Fever', 'min_value': 95, 'max_value': 105, 'unit': '°F'},
            {'name': 'Headache', 'min_value': 0, 'max_value': 10, 'unit': 'scale'},
            {'name': 'Cough', 'min_value': 0, 'max_value': 10, 'unit': 'scale'},
            {'name': 'Fatigue', 'min_value': 0, 'max_value': 10, 'unit': 'scale'},
            {'name': 'Body Pain', 'min_value': 0, 'max_value': 10, 'unit': 'scale'},
        ]
        
        for symptom_data in symptoms:
            Symptom.objects.create(**symptom_data)
        
        # Create diseases
        diseases = [
            {'name': 'Common Cold', 'description': 'Viral infection of the upper respiratory tract'},
            {'name': 'Flu', 'description': 'Influenza viral infection affecting the respiratory system'},
            {'name': 'Malaria', 'description': 'Mosquito-borne infectious disease causing fever and chills'},
            {'name': 'Dengue', 'description': 'Mosquito-borne tropical disease causing high fever'},
        ]
        
        disease_objs = {}
        for disease_data in diseases:
            disease_objs[disease_data['name']] = Disease.objects.create(**disease_data)
        
        # Create rules
        rules = [
            # Common Cold rules
            {'disease': 'Common Cold', 'symptom': 'Fever', 'severity': 'low', 'weight': 0.7},
            {'disease': 'Common Cold', 'symptom': 'Headache', 'severity': 'low', 'weight': 0.6},
            {'disease': 'Common Cold', 'symptom': 'Cough', 'severity': 'medium', 'weight': 0.8},
            
            # Flu rules
            {'disease': 'Flu', 'symptom': 'Fever', 'severity': 'medium', 'weight': 0.9},
            {'disease': 'Flu', 'symptom': 'Headache', 'severity': 'high', 'weight': 0.8},
            {'disease': 'Flu', 'symptom': 'Body Pain', 'severity': 'high', 'weight': 0.9},
            
            # Malaria rules
            {'disease': 'Malaria', 'symptom': 'Fever', 'severity': 'high', 'weight': 1.0},
            {'disease': 'Malaria', 'symptom': 'Headache', 'severity': 'medium', 'weight': 0.7},
            {'disease': 'Malaria', 'symptom': 'Fatigue', 'severity': 'high', 'weight': 0.8},
            
            # Dengue rules
            {'disease': 'Dengue', 'symptom': 'Fever', 'severity': 'high', 'weight': 1.0},
            {'disease': 'Dengue', 'symptom': 'Headache', 'severity': 'high', 'weight': 0.9},
            {'disease': 'Dengue', 'symptom': 'Body Pain', 'severity': 'high', 'weight': 0.9},
        ]
        
        for rule_data in rules:
            DiseaseRule.objects.create(
                disease=disease_objs[rule_data['disease']],
                symptom=Symptom.objects.get(name=rule_data['symptom']),
                severity=rule_data['severity'],
                weight=rule_data['weight']
            )
        
        return redirect('fuzzy_app:index')  # Added app namespace
    
    return redirect('fuzzy_app:index')  # Added app namespace