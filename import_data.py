import pandas as pd
from fuzzy_app.models import Symptom, Disease, DiseaseRule

def import_from_csv(csv_path):
    # First clear existing data to avoid conflicts
    DiseaseRule.objects.all().delete()
    Symptom.objects.all().delete()
    Disease.objects.all().delete()
    
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        try:
            # 1. Create Disease
            disease, _ = Disease.objects.get_or_create(
                name=row['disease'],
                defaults={'description': f"Imported {row['disease']}"}
            )
            
            # 2. Process Symptoms
            for i in range(1, 6):
                col_name = f'symptom_{i}'
                if pd.notna(row.get(col_name, '')) and str(row[col_name]).strip() != '':
                    if ',' in str(row[col_name]):  # Check if value contains a comma
                        name, severity = str(row[col_name]).split(',', 1)
                        name = name.strip()
                        severity = severity.strip()
                        
                        symptom, _ = Symptom.objects.get_or_create(
                            name=name,
                            defaults={'min_value': 0, 'max_value': 10}
                        )
                        
                        # 3. Create Rules with error handling
                        try:
                            severity_level = 'high' if float(severity) > 0.7 else 'medium'
                            DiseaseRule.objects.get_or_create(
                                disease=disease,
                                symptom=symptom,
                                severity=severity_level,
                                defaults={'weight': float(severity)}
                            )
                        except ValueError as e:
                            print(f"Error processing severity for {name}: {e}")
                            continue
        except Exception as e:
            print(f"Error processing row {_}: {e}")
            continue

    print("Data import completed successfully!")
    print(f"Total Diseases: {Disease.objects.count()}")
    print(f"Total Symptoms: {Symptom.objects.count()}")
    print(f"Total Rules: {DiseaseRule.objects.count()}")

if __name__ == "__main__":
    import_from_csv('medical_data_1000.csv')