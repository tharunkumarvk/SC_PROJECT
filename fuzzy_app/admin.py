from django.contrib import admin
from .models import Symptom, Disease, DiseaseRule

class DiseaseRuleInline(admin.TabularInline):
    model = DiseaseRule
    extra = 1

class DiseaseAdmin(admin.ModelAdmin):
    inlines = [DiseaseRuleInline]

admin.site.register(Symptom)
admin.site.register(Disease, DiseaseAdmin)