"""
Dynamic symptom input form.

Generates one numeric field per symptom from the knowledge base,
organized by SYMPTOM_CATEGORIES for a clean grouped UI.
"""

from django import forms
from .tropical_diseases import SYMPTOMS, SYMPTOM_ORDER, SYMPTOM_CATEGORIES


class SymptomForm(forms.Form):
    """Form with one optional numeric field per symptom."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in SYMPTOM_ORDER:
            info = SYMPTOMS[key]
            self.fields[key] = forms.FloatField(
                label=info['name'],
                required=False,
                min_value=info['min'],
                max_value=info['max'],
                help_text=info['help'],
                widget=forms.NumberInput(attrs={
                    'class': 'form-control',
                    'placeholder': f"{info['min']}–{info['max']} {info['unit']}",
                    'step': '0.1' if key == 'fever' else '1',
                }),
            )

    def get_symptom_values(self):
        """Return dict of provided (non-None) symptom values."""
        if not self.is_valid():
            return {}
        return {
            key: val
            for key, val in self.cleaned_data.items()
            if val is not None
        }

    def get_grouped_fields(self):
        """Yield (category_name, [fields]) for template rendering."""
        for cat_name, keys in SYMPTOM_CATEGORIES.items():
            fields = [self[k] for k in keys if k in self.fields]
            if fields:
                yield cat_name, fields
