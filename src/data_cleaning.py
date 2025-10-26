import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import sys
from typing import Tuple, List, Union

# --- 1. Column Name Cleaning ---
def clean_col_name(col: str) -> str:
    """Standardizes column names to snake_case."""
    col = col.strip().lower()
    col = col.replace(" ", "_")
    col = col.replace("-", "_")
    col = col.replace("(", "")
    col = col.replace(")", "")
    col = col.replace(".", "")
    # Remove double underscores that might result from cleaning, e.g., 'fluid_overload_1'
    col = "_".join(filter(None, col.split("_"))) 
    return col

# --- 2. Main Data Loading and Cleaning Function ---
def load_and_clean_data(
    file_path: str
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, LabelEncoder]:
    """
    Loads the raw disease data, performs all necessary cleaning (including NaN handling 
    and rogue string removal), and encodes the target variable.

    Returns: X (features), y_encoded (encoded target), df (cleaned full data), le (LabelEncoder)
    """
    data_loaded = False
    df = None 
    
    # --- Load Data with Encoding Fallback ---
    try:
        df = pd.read_csv(file_path)
        data_loaded = True
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin-1')
            data_loaded = True
        except FileNotFoundError:
            print(f"Error: {file_path} not found. Ensure it is in the 'data/' directory.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred during file loading: {e}")
            sys.exit(1)
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Ensure it is in the 'data/' directory.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during file loading: {e}")
        sys.exit(1)

    if not data_loaded:
        # Should be caught by sys.exit(1) above, but kept as a safeguard
        raise RuntimeError("Data loading failed and script should have exited.")

    # Apply column cleaning to the entire DataFrame
    df.columns = [clean_col_name(col) for col in df.columns]
    
    # CRITICAL CLEANING STEP 1: Replace all standard NaNs (floats) with 0
    df = df.fillna(0) 

    # --- Separate features (X) and target (y) ---
    if 'prognosis' not in df.columns:
        print("Error: 'prognosis' column not found after cleaning. Check your dataset structure.")
        sys.exit(1)
    
    y = df['prognosis']
    X = df.drop('prognosis', axis=1)
    
    # CRITICAL CLEANING STEP 2: Drop purely empty columns (where all values are 0)
    X = X.loc[:, (X != 0).any(axis=0)]
    
    # CRITICAL CLEANING STEP 3 (The fix for IntCastingNaNError): 
    # Coerce any remaining non-numeric string values (like 'N/A' or '-') to NaN,
    # then fill those newly created NaNs with 0.
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)
    
    # --- Encode the Target Variable (y) ---
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # --- Final Feature Downcasting ---
    # Downcast X to the memory-efficient integer type
    X = X.astype(np.int8)

    print(f"Data successfully cleaned and prepared. Features shape: {X.shape}, Target samples: {len(y_encoded)}")
    
    return X, y_encoded, df, le

if __name__ == '__main__':
    # Simple test case for the cleaning function (requires 'data/disease_data.csv')
    try:
        data_path = "data/disease_train_data.csv"
        print(f"Running data_cleaning.py as main to test functionality on {data_path}...")
        X, y_encoded, df_cleaned, le = load_and_clean_data(data_path)
        print("Test complete. Data preparation succeeded.")
    except Exception as e:
        print(f"Test failed: {e}")

