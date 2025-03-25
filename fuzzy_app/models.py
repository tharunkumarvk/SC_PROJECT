from django.db import models

class Symptom(models.Model):
    name = models.CharField(max_length=100, unique=True)
    min_value = models.FloatField(default=0)
    max_value = models.FloatField(default=10)
    unit = models.CharField(max_length=20, blank=True)

    def __str__(self):
        return self.name

class Disease(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True)

    def __str__(self):
        return self.name

class DiseaseRule(models.Model):
    disease = models.ForeignKey(Disease, on_delete=models.CASCADE)
    symptom = models.ForeignKey(Symptom, on_delete=models.CASCADE)
    severity = models.CharField(max_length=20, choices=[
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High')
    ])
    weight = models.FloatField(default=1.0)

    class Meta:
        unique_together = ('disease', 'symptom', 'severity')

    def __str__(self):
        return f"{self.disease.name} - {self.symptom.name} ({self.severity})"