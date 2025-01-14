import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 1. Load and prepare data
def prepare_data(filepath):
    # Read the data
    df = pd.read_csv(filepath)
    
    # Convert gender to numerical (0 for F, 1 for M)
    df['Gender'] = (df['Gender'] == 'M').astype(int)
    
    # Split diagnoses that have multiple conditions
    diagnoses = df['Diagnosis'].str.split('; ')
    
    # Use MultiLabelBinarizer for diagnoses
    mlb = MultiLabelBinarizer()
    diagnosis_matrix = mlb.fit_transform(diagnoses)
    
    # Create DataFrame with binary diagnosis columns
    diagnosis_df = pd.DataFrame(diagnosis_matrix, columns=mlb.classes_)
    
    # Separate features and target
    X = df.drop(['Diagnosis'], axis=1)
    y = diagnosis_df
    
    return X, y, mlb

# 2. Data preprocessing
def preprocess_data(X_train, X_test):
    # Scale the numerical features
    scaler = StandardScaler()
    
    # Get column names for later use
    columns = X_train.columns
    
    # Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=columns)
    
    return X_train_scaled, X_test_scaled, scaler

# 3. Train model
def train_model(X_train, y_train):
    # Initialize Random Forest Classifier with multi-output support
    rf_model = RandomForestClassifier(n_estimators=100, 
                                    max_depth=10,
                                    random_state=42)
    
    # Train the model
    rf_model.fit(X_train, y_train)
    
    return rf_model

# 4. Evaluate model
def evaluate_model(model, X_test, y_test, mlb):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print classification report for each diagnosis type
    print("\nClassification Reports by Condition:")
    for i, diagnosis in enumerate(mlb.classes_):
        print(f"\n{diagnosis}:")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
    
    # Calculate overall accuracy
    accuracy = (y_test == y_pred).all(axis=1).mean()
    print(f"\nOverall Exact Match Accuracy: {accuracy:.2f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return feature_importance

# 5. Function to make new predictions
def predict_diagnosis(model, scaler, mlb, new_data):
    # Scale the new data
    new_data_scaled = scaler.transform(new_data)
    
    # Make prediction
    prediction_prob = model.predict_proba(new_data_scaled)
    prediction = model.predict(new_data_scaled)
    
    # Convert binary predictions to diagnosis labels
    diagnoses = []
    probabilities = []
    
    for i, row in enumerate(prediction):
        row_diagnoses = []
        row_probs = []
        for j, val in enumerate(row):
            if val == 1:
                row_diagnoses.append(mlb.classes_[j])
            row_probs.append(prediction_prob[j][i][1])  # probability of positive class
        diagnoses.append('; '.join(row_diagnoses) if row_diagnoses else 'Fit')
        probabilities.append(dict(zip(mlb.classes_, row_probs)))
    
    return diagnoses, probabilities

# Function to make predictions for new patients
def predict_for_new_patient(model, scaler, mlb, 
                          gender, hemoglobin, blood_sugar, 
                          bp_systolic, bp_diastolic, 
                          total_cholesterol, heart_rate, 
                          creatinine, rbc):
    """
    Make predictions for a new patient based on their health markers.
    Gender should be 'M' or 'F'.
    """
    # Convert gender to numerical (M=1, F=0)
    gender_num = 1 if gender == 'M' else 0
    
    # Create DataFrame with patient data
    patient_data = pd.DataFrame({
        'Gender': [gender_num],
        'Hemoglobin': [hemoglobin],
        'Blood_Sugar': [blood_sugar],
        'BP_Systolic': [bp_systolic],
        'BP_Diastolic': [bp_diastolic],
        'Total_Cholesterol': [total_cholesterol],
        'Heart_Rate': [heart_rate],
        'Creatinine': [creatinine],
        'RBC': [rbc]
    })
    
    # Make prediction
    diagnoses, probabilities = predict_diagnosis(model, scaler, mlb, patient_data)
    
    return diagnoses[0], probabilities[0]

# Main execution
if __name__ == "__main__":
    # 1. Load and prepare data
    print("Loading and preparing data...")
    X, y, mlb = prepare_data('health_markers_dataset4.csv')
    
    # 2. Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42)
    
    # 3. Preprocess the data
    print("Preprocessing data...")
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    
    # 4. Train the model
    print("Training model...")
    model = train_model(X_train_scaled, y_train)
    
    # 5. Evaluate the model
    print("Evaluating model...")
    feature_importance = evaluate_model(model, X_test_scaled, y_test, mlb)
    
    # 6. Example prediction
    print("\nExample Prediction:")
    diagnosis, probabilities = predict_for_new_patient(
        model, scaler, mlb,
        gender='M',
        hemoglobin=14.5,
        blood_sugar=350,  
        bp_systolic=120,
        bp_diastolic=80,
        total_cholesterol=180,
        heart_rate=75,
        creatinine=1.1,
        rbc=5.0
    )
    
    print(f"\nPredicted Diagnosis: {diagnosis}")
    print("\nProbability for each condition:")
    for condition, prob in probabilities.items():
        print(f"{condition}: {prob:.2f}")