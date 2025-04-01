import csv
import random
import os
from faker import Faker

# Configure output path
output_path = os.path.join(os.path.expanduser('~'), 'Documents', 'medical_data_1000.csv')
fake = Faker()

# Disease-symptom mapping (minimum 5 symptoms each)
diseases = {
    "Common Cold": ["Cough", "Sore Throat", "Runny Nose", "Sneezing", "Mild Fever", "Congestion"],
    "Influenza": ["High Fever", "Body Aches", "Fatigue", "Headache", "Chills", "Dry Cough"],
    "Pneumonia": ["High Fever", "Cough with Phlegm", "Chest Pain", "Shortness of Breath", "Fatigue", "Confusion"],
    "Asthma": ["Wheezing", "Shortness of Breath", "Chest Tightness", "Coughing at Night", "Rapid Breathing"],
    "COVID-19": ["Fever", "Dry Cough", "Fatigue", "Loss of Taste", "Shortness of Breath"],
    "Bronchitis": ["Persistent Cough", "Wheezing", "Chest Discomfort", "Mild Fever", "Fatigue"],
    "Strep Throat": ["Sore Throat", "Fever", "Swallowing Pain", "Swollen Tonsils", "Headache"],
    "Allergies": ["Sneezing", "Itchy Eyes", "Runny Nose", "Congestion", "Cough"],
    "Migraine": ["Headache", "Nausea", "Sensitivity to Light", "Sensitivity to Sound"],
    "Diabetes": ["Frequent Urination", "Increased Thirst", "Blurred Vision", "Fatigue", "Slow Healing"]
}

def get_symptoms(disease):
    """Safely get 3-5 symptoms"""
    available = diseases[disease]
    k = min(random.randint(3, 5), len(available))
    return random.sample(available, k)

# Generate and save data
try:
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "age", "gender", "disease"] + [f"symptom_{i}" for i in range(1,6)])
        
        for _ in range(1000):
            disease = random.choice(list(diseases.keys()))
            symptoms = get_symptoms(disease)
            symptom_data = [f"{s},{round(random.uniform(0.3, 1.0), 1)}" for s in symptoms]
            symptom_data += [""] * (5 - len(symptoms))  # Pad empty columns
            
            writer.writerow([
                fake.uuid4(),
                random.randint(1, 90),
                random.choice(["M", "F"]),
                disease,
                *symptom_data
            ])
    
    print(f"Success! File saved to: {output_path}")
    print(f"First 3 rows preview:")
    with open(output_path, 'r') as f:
        for _ in range(4):  # Header + 3 rows
            print(next(f).strip())

except PermissionError:
    print(f"ERROR: Still can't write to {output_path}. Try:")
    print("1. Close any programs using Documents folder")
    print("2. Temporarily disable antivirus")
    print(f"3. Manually create empty file first at: {output_path}")
except Exception as e:
    print(f"Unexpected error: {str(e)}")