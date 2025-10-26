
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib 
import os
import json
import sys

# --- Configuration ---
FILE_PATH = "data/disease_train_data.csv"
MODEL_DIR = 'models'

def clean_col_name(col):
    """Standardizes column names to snake_case."""
    col = col.strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace(".", "")
    col = "_".join(filter(None, col.split("_")))
    return col

def train_and_save_model():
    """
    Executes the full machine learning training pipeline using ALL data.
    With such a small dataset (55 samples, 54 diseases), we use all data for training.
    """
    # Ensure the models directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Load Data ---
    try:
        df = pd.read_csv(FILE_PATH)
        print(f"Data loaded successfully. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {FILE_PATH} not found.")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Apply column cleaning
    df.columns = [clean_col_name(col) for col in df.columns]
    
    # Handle missing values
    df = df.fillna(0)

    # Separate features and target
    if 'prognosis' not in df.columns:
        print("Error: 'prognosis' column not found.")
        return
    
    y = df['prognosis']
    X = df.drop('prognosis', axis=1)
    
    # Drop empty columns and handle non-numeric values
    X = X.loc[:, (X != 0).any(axis=0)]
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    X = X.astype(np.int8)

    # Encode target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Get symptom columns
    symptom_columns = X.columns.tolist()

    print(f"Number of features: {len(symptom_columns)}")
    print(f"Number of diseases: {len(le.classes_)}")

    # Train model with balanced class weight using ALL data
    print("Training model with all data...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y_encoded)
    
    # Save model assets
    MODEL_FILENAME = os.path.join(MODEL_DIR, 'disease_model.pkl') 
    ENCODER_FILENAME = os.path.join(MODEL_DIR, 'label_encoder.pkl') 
    SYMPTOMS_FILENAME = os.path.join(MODEL_DIR, 'symptoms.json') 
    
    try:
        joblib.dump(model, MODEL_FILENAME)
        joblib.dump(le, ENCODER_FILENAME)
        
        with open(SYMPTOMS_FILENAME, 'w') as f:
            json.dump(symptom_columns, f, indent=2)
        
        print(f"✓ Model assets saved to '{MODEL_DIR}/'")
        
        # Print disease distribution
        unique_diseases, counts = np.unique(y, return_counts=True)
        print(f"\nTraining data distribution:")
        for disease, count in zip(unique_diseases, counts):
            print(f"  {disease}: {count} samples")
            
    except Exception as e:
        print(f"✗ Error saving files: {e}")
        return
    
    # Test prediction with multiple symptom combinations
    print("\n" + "="*50)
    print("TESTING PREDICTIONS WITH DIFFERENT SYMPTOMS")
    print("="*50)
    
    test_cases = [
        ["itching", "skin_rash", "nodal_skin_eruptions"],  # Should predict Fungal infection
        ["continuous_sneezing", "shivering", "chills"],    # Should predict Allergy
        ["mood_swings", "weight_loss", "lethargy", "indigestion", "runny_nose", "weakness_in_limbs", "bloody_stool"],  # Your original test case
        ["high_fever", "headache", "vomiting"],            # Should predict something else
    ]
    
    for i, test_symptoms in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i}: {test_symptoms} ---")
        
        feature_vector = np.zeros(len(symptom_columns), dtype=np.int8)
        
        matched_count = 0
        matched_symptoms = []
        for j, symptom in enumerate(symptom_columns):
            if symptom in test_symptoms:
                feature_vector[j] = 1
                matched_count += 1
                matched_symptoms.append(symptom)
        
        print(f"Matched {matched_count} symptoms: {matched_symptoms}")
        
        if feature_vector.sum() > 0:
            X_test_sample = feature_vector.reshape(1, -1)
            probabilities = model.predict_proba(X_test_sample)[0]
            
            # Get top 5 predictions
            top_5_indices = np.argsort(probabilities)[-5:][::-1]
            
            print("Top 5 predictions:")
            for idx in top_5_indices:
                disease = le.classes_[idx]
                prob = probabilities[idx]
                if prob > 0.01:  # Only show predictions with >1% probability
                    print(f"  {disease}: {prob:.4f} ({prob*100:.2f}%)")
        else:
            print("No symptoms matched!")
    
    # Calculate training accuracy
    train_predictions = model.predict(X)
    train_accuracy = np.mean(train_predictions == y_encoded)
    print(f"\n" + "="*50)
    print(f"TRAINING COMPLETE - Accuracy: {train_accuracy:.2%}")
    print("="*50)
    
    # Show some statistics
    print(f"\nModel Statistics:")
    print(f"- Total samples: {len(y)}")
    print(f"- Total diseases: {len(le.classes_)}")
    print(f"- Total symptoms: {len(symptom_columns)}")
    print(f"- Average samples per disease: {len(y)/len(le.classes_):.2f}")

if __name__ == '__main__':
    train_and_save_model()