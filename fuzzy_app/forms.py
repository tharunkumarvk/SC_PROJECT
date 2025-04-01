from django import forms
from .models import Symptom

class SymptomForm(forms.Form):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for symptom in Symptom.objects.all():
            self.fields[f'symptom_{symptom.id}'] = forms.FloatField(
                label=symptom.name,
                required=False,
                min_value=symptom.min_value,
                max_value=symptom.max_value,
                widget=forms.NumberInput(attrs={
                    'class': 'form-control',
                    'placeholder': f'Enter {symptom.name} ({symptom.unit})'
                })
            )