
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

# --- 1. DEFINE PATHS (Adjust these if necessary) ---
MODEL_PATH = 'models/disease_model.pkl'
# IMPORTANT: Replace 'data/Training.csv' with the actual path to your training data file.
TRAINING_DATA_PATH = 'data/disease_train_data.csv'

# Ensure the models directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)


def retrain_and_save_model():
    """
    Loads the training data, re-initializes and trains the required ML objects, 
    and saves them back to the PKL file using the CURRENT scikit-learn version.
    """
    print("--- Starting model retraining and saving process ---")
    
    try:
        # Load your training dataset
        df = pd.read_csv(TRAINING_DATA_PATH)

        # --- Data Preparation Steps (Crucial: Replace with your actual logic) ---
        # Assuming the last column is the target (Disease) and others are Symptoms
        X = df.iloc[:, :-1]  # Symptoms features
        y = df.iloc[:, -1]   # Target (Disease)

        # 1. Label Encoder (for converting disease names to numbers)
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print("LabelEncoder fitted successfully.")

        # If your features (X) are not yet numeric (e.g., Symptom_1, Symptom_2, etc.), 
        # you need to apply one-hot encoding or similar transformation here.
        # Since the model expects a specific numeric format, ensure your X data
        # matches the format used for the original training.
        
        # --- Model Training ---
        
        # 2. Decision Tree Classifier
        dtc = DecisionTreeClassifier()
        dtc.fit(X, y_encoded)
        print("DecisionTreeClassifier trained successfully.")

        # 3. Random Forest Classifier
        rfc = RandomForestClassifier()
        rfc.fit(X, y_encoded)
        print("RandomForestClassifier trained successfully.")

        # --- Model Saving ---
        # Save all required components (models and encoder) into a dictionary
        model_assets = {
            'dtc_model': dtc,
            'rfc_model': rfc,
            'label_encoder': le
        }
        
        joblib.dump(model_assets, MODEL_PATH)
        print(f"SUCCESS: Model assets re-saved to {MODEL_PATH} using scikit-learn v{pd.__version__}.")

    except FileNotFoundError:
        print(f"ERROR: Training data not found at {TRAINING_DATA_PATH}. Please update the path inside fix_model_save.py.")
    except Exception as e:
        print(f"An unexpected error occurred during training or saving: {e}")

if __name__ == '__main__':
    retrain_and_save_model()
