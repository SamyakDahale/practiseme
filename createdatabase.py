import pandas as pd
import numpy as np
import random

def generate_health_data(n_people=25000):
    # Initialize empty dataframe
    df = pd.DataFrame()
    
    # Generate gender
    df['gender'] = np.random.choice(['M', 'F'], size=n_people)
    
    # Generate normal values for all markers (70% of population)
    normal_count = int(n_people * 0.70)
    single_critical_count = int(n_people * 0.25)
    double_critical_count = int(n_people * 0.05)
    
    def generate_hemoglobin(gender):
        if gender == 'M':
            return np.random.uniform(13.8, 17.2)
        return np.random.uniform(12.1, 15.1)
    
    def generate_critical_hemoglobin(gender):
        return np.random.uniform(3, 7)  # Only low values for anemia
    
    # Generate base values for everyone
    df['hemoglobin'] = df['gender'].apply(generate_hemoglobin)
    df['blood_sugar'] = np.random.uniform(70, 140, n_people)
    df['systolic_bp'] = np.random.uniform(90, 120, n_people)
    df['cholesterol'] = np.random.uniform(150, 200, n_people)
    df['heart_rate'] = np.random.uniform(60, 100, n_people)
    df['oxygen_saturation'] = np.random.uniform(95, 100, n_people)
    df['creatinine'] = df['gender'].apply(lambda x: np.random.uniform(0.7, 1.3) if x == 'M' else np.random.uniform(0.6, 1.1))
    df['crp'] = np.random.uniform(0, 10, n_people)
    
    # Function to make one marker critical
    def make_one_critical(row):
        marker = np.random.choice(['hb', 'bs', 'bp', 'chol', 'hr', 'o2', 'cr', 'crp'])
        if marker == 'hb':
            row['hemoglobin'] = generate_critical_hemoglobin(row['gender'])
        elif marker == 'bs':
            row['blood_sugar'] = np.random.uniform(300, 400)
        elif marker == 'bp':
            if np.random.random() < 0.5:
                row['systolic_bp'] = np.random.uniform(180, 200)  # High BP
            else:
                row['systolic_bp'] = np.random.uniform(60, 90)    # Low BP
        elif marker == 'chol':
            row['cholesterol'] = np.random.uniform(240, 300)
        elif marker == 'hr':
            row['heart_rate'] = np.random.choice([np.random.uniform(20, 40), np.random.uniform(120, 150)])
        elif marker == 'o2':
            row['oxygen_saturation'] = np.random.uniform(80, 90)
        elif marker == 'cr':
            row['creatinine'] = np.random.uniform(5, 8)
        elif marker == 'crp':
            row['crp'] = np.random.uniform(100, 150)
        return row
    
    # Apply critical values
    critical_indices = np.random.choice(df.index, size=single_critical_count + double_critical_count, replace=False)
    
    # Single critical
    single_critical_indices = critical_indices[:single_critical_count]
    for idx in single_critical_indices:
        df.loc[idx] = make_one_critical(df.loc[idx])
    
    # Double critical
    double_critical_indices = critical_indices[single_critical_count:]
    for idx in double_critical_indices:
        df.loc[idx] = make_one_critical(df.loc[idx])
        df.loc[idx] = make_one_critical(df.loc[idx])
    
    # Add diagnosis
    def get_diagnosis(row):
        diagnoses = []
        if row['hemoglobin'] < 7:
            diagnoses.append('anemia')
        if row['blood_sugar'] > 300:
            diagnoses.append('diabetes')
        if row['systolic_bp'] > 180 or row['systolic_bp'] < 90:
            diagnoses.append('hypertension')
        if row['cholesterol'] > 240:
            diagnoses.append('hypercholesterolemia')
        if row['heart_rate'] < 40 or row['heart_rate'] > 120:
            diagnoses.append('bradycardia')
        if row['oxygen_saturation'] < 90:
            diagnoses.append('asthma')
        if row['creatinine'] > 5:
            diagnoses.append('kidney failure')
        if row['crp'] > 100:
            diagnoses.append('pneumonia')
        
        return ', '.join(diagnoses) if diagnoses else 'fit'
    
    df['diagnosis'] = df.apply(get_diagnosis, axis=1)
    
    # Round all values to 1 decimal place
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(1)
    
    return df

# Generate the dataset
health_data = generate_health_data()

# Save to CSV
health_data.to_csv('health_markers_dataset_new.csv', index=False)