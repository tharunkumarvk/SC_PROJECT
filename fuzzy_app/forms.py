from django import forms
from .models import Symptom

class SymptomInputForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        symptoms = Symptom.objects.all()
        
        for symptom in symptoms:
            field_name = f"symptom_{symptom.id}"
            self.fields[field_name] = forms.FloatField(
                label=f"{symptom.name} ({symptom.unit})",
                required=False,
                widget=forms.NumberInput(attrs={
                    'class': 'form-control',
                    'min': symptom.min_value,
                    'max': symptom.max_value,
                    'step': '0.1',
                    'placeholder': f'Enter {symptom.name} value'
                }),
                min_value=symptom.min_value,
                max_value=symptom.max_value
            )