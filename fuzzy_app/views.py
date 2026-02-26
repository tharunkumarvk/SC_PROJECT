"""
Views for the Tropical Disease Prediction app.
"""

from django.shortcuts import render
from .forms import SymptomForm
from .consensus import consensus_predict
from .tropical_diseases import DISEASE_PROFILES, SYMPTOMS, SYMPTOM_ORDER


def index(request):
    """Main prediction page — shows form, handles prediction on POST."""
    if request.method == 'POST':
        form = SymptomForm(request.POST)
        if form.is_valid():
            symptom_values = form.get_symptom_values()
            if symptom_values:
                result = consensus_predict(symptom_values)

                # Build display list of provided symptoms
                display_symptoms = []
                for key in SYMPTOM_ORDER:
                    if key in symptom_values:
                        info = SYMPTOMS[key]
                        display_symptoms.append({
                            'name': info['name'],
                            'value': symptom_values[key],
                            'unit': info['unit'],
                        })

                return render(request, 'fuzzy_app/results.html', {
                    'result': result,
                    'display_symptoms': display_symptoms,
                })
    else:
        form = SymptomForm()

    return render(request, 'fuzzy_app/index.html', {
        'form': form,
        'disease_count': len(DISEASE_PROFILES),
        'grouped_fields': list(form.get_grouped_fields()),
    })


def about(request):
    """About page with system information and disease details."""
    diseases = []
    for name, profile in DISEASE_PROFILES.items():
        diseases.append({
            'name': name,
            'description': profile['description'],
            'hallmarks': profile.get('hallmarks', []),
            'references': profile.get('references', []),
        })
    return render(request, 'fuzzy_app/about.html', {
        'diseases': diseases,
        'symptom_count': len(SYMPTOM_ORDER),
    })
