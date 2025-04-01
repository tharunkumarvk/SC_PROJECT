from django.db import models

class Symptom(models.Model):
    name = models.CharField(max_length=100, unique=True)
    min_value = models.FloatField(default=0)
    max_value = models.FloatField(default=10)
    unit = models.CharField(max_length=20, default='severity (0-10)')

    def __str__(self):
        return f"{self.name} ({self.unit})"

class Disease(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    precautions = models.TextField(blank=True)

    def __str__(self):
        return self.name

class DiseaseRule(models.Model):
    SEVERITY_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High')
    ]
    
    disease = models.ForeignKey(Disease, on_delete=models.CASCADE)
    symptom = models.ForeignKey(Symptom, on_delete=models.CASCADE)
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    weight = models.FloatField(default=1.0)
    threshold_min = models.FloatField(default=0)
    threshold_max = models.FloatField(default=10)

    class Meta:
        unique_together = ('disease', 'symptom', 'severity')

    def __str__(self):
        return f"{self.disease.name}: {self.symptom.name} ({self.severity})"