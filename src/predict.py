
import pandas as pd
import joblib
import json
import os
import numpy as np

# --- Configuration & Global Variables ---
MODEL_FILE = os.path.join('models', 'disease_model.pkl')
ENCODER_FILE = os.path.join('models', 'label_encoder.pkl')
SYMPTOM_JSON = 'models/symptoms.json'

# Global variables to hold model assets once loaded
model = None
label_encoder = None
FULL_SYMPTOM_LIST = None

def clean_symptom_name(symptom_name: str) -> str:
    """Converts symptom names to standardized format."""
    symptom_name = symptom_name.strip().lower()
    symptom_name = symptom_name.replace(" ", "_")
    symptom_name = symptom_name.replace("-", "_")
    symptom_name = symptom_name.replace("(", "")
    symptom_name = symptom_name.replace(")", "")
    symptom_name = symptom_name.replace(".", "")
    symptom_name = "_".join(filter(None, symptom_name.split("_")))
    return symptom_name

def load_model_assets(model_path=MODEL_FILE, encoder_path=ENCODER_FILE, symptom_json_path=SYMPTOM_JSON):
    """
    Loads the trained model assets (model, label encoder, and symptom list)
    from disk into global variables.
    """
    global model, label_encoder, FULL_SYMPTOM_LIST

    print(f"Attempting to load model from: {model_path}")

    try:
        # Load the trained model and label encoder separately
        model = joblib.load(model_path)
        label_encoder = joblib.load(encoder_path)
        
        # Load the canonical list of all possible symptoms
        with open(symptom_json_path, 'r') as f:
            symptoms_list = json.load(f)
        
        FULL_SYMPTOM_LIST = symptoms_list

        if not FULL_SYMPTOM_LIST or not isinstance(FULL_SYMPTOM_LIST, list):
            raise ValueError(f"Symptom list loaded incorrectly. Size: {len(FULL_SYMPTOM_LIST) if FULL_SYMPTOM_LIST else 0}")
            
        print(f"SUCCESS: Model assets loaded. Total symptoms found: {len(FULL_SYMPTOM_LIST)}")
        print(f"Number of diseases: {len(label_encoder.classes_)}")
        
        return True

    except FileNotFoundError as e:
        print(f"ERROR: Model or data file not found: {e}")
        raise e
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse symptom list JSON. Check file format. Details: {e}")
        raise e
    except Exception as e:
        print(f"ERROR: An unexpected error occurred during model loading: {e}")
        raise e

def get_feature_vector(selected_symptoms):
    """
    Converts a list of selected symptom strings into a numeric 
    feature vector (1s for selected, 0s for not selected).
    """
    if FULL_SYMPTOM_LIST is None:
        print("ERROR: Symptom list is not loaded.")
        return None

    # Initialize the feature vector with zeros
    feature_vector = np.zeros(len(FULL_SYMPTOM_LIST), dtype=np.int32)
    
    # Convert user symptoms to match the format in FULL_SYMPTOM_LIST
    selected_symptoms_clean = [clean_symptom_name(symptom) for symptom in selected_symptoms]
    
    print(f"DEBUG: User symptoms cleaned: {selected_symptoms_clean}")
    
    # Find the index of each selected symptom and set the value to 1
    matched_count = 0
    for i, symptom in enumerate(FULL_SYMPTOM_LIST):
        if symptom in selected_symptoms_clean:
            feature_vector[i] = 1
            matched_count += 1
            
    print(f"DEBUG: Matched {matched_count} symptoms out of {len(selected_symptoms)} provided")
    print(f"DEBUG: Available symptoms sample: {FULL_SYMPTOM_LIST[:5]}...")
    
    return feature_vector.reshape(1, -1)

def predict_disease(selected_symptoms):
    """
    Takes selected symptoms, converts them to a feature vector, and predicts 
    the top N possible diseases and their confidence levels.
    """
    global model, label_encoder, FULL_SYMPTOM_LIST

    if model is None or label_encoder is None or FULL_SYMPTOM_LIST is None:
        try:
            load_model_assets()
        except Exception:
            return "Error: Model not loaded. Check server logs.", []

    if not selected_symptoms:
        return "No Symptoms Selected", []

    # Get the correctly formatted feature vector
    X_test = get_feature_vector(selected_symptoms)
    if X_test is None:
        return "Error: Feature vector creation failed.", []

    try:
        # Get probability scores for all classes
        probabilities = model.predict_proba(X_test)[0]

        # Map probabilities back to disease names
        disease_names = label_encoder.classes_
        results = []
        
        for i, prob in enumerate(probabilities):
            results.append({
                'disease': disease_names[i],
                'confidence': round(prob * 100, 2)
            })

        # Sort results by confidence (descending)
        results.sort(key=lambda x: x['confidence'], reverse=True)

        # Identify the primary prediction
        primary_prediction = results[0]['disease'] if results else "Unknown"
        
        # Filter out predictions with very low confidence and return top 3
        final_results = [res for res in results if res['confidence'] > 0.1]
        
        print(f"DEBUG: Primary prediction: {primary_prediction}")
        print(f"DEBUG: Top 3 results: {[f'{r['disease']}: {r['confidence']}%' for r in final_results[:3]]}")
        
        return primary_prediction, final_results[:5]  # Return top 5 results
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Error: Prediction failed.", []