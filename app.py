
import json
from flask import Flask, request, jsonify, render_template
import os
import sys

# Define base directory and add 'src' to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)

# Import as module for safer access to globals
import src.predict as predictor_module

# --- Load Model Assets Globally ---
# This block runs once when the server starts.
try:
    # 1. Load assets
    predictor_module.load_model_assets()
    
    # 2. Check if loading was truly successful before proceeding
    if (predictor_module.FULL_SYMPTOM_LIST is None or 
        predictor_module.model is None or 
        predictor_module.label_encoder is None):
        raise Exception("Global variables were not populated after load_model_assets completed.")

    print("SERVER READY: Model assets loaded successfully for API use.")
    print(f"Model type: {type(predictor_module.model)}")
    print(f"Label encoder classes: {len(predictor_module.label_encoder.classes_)}")
    print(f"Symptoms count: {len(predictor_module.FULL_SYMPTOM_LIST)}")

except Exception as e:
    print(f"FATAL ERROR: Failed to load model assets. Server cannot start. Error: {e}")
    sys.exit(1)


# --- API Route to Get Initial Symptom List (For the Website) ---
@app.route('/symptoms', methods=['GET'])
def get_symptoms():
    """
    Returns the ordered list of all symptoms needed by the front-end,
    converting machine-readable names to human-readable titles.
    """
    # Access global list via the module reference
    if predictor_module.FULL_SYMPTOM_LIST is None:
        return jsonify({'error': 'Symptom data not initialized on server.'}), 500
        
    # Convert machine-readable names (e.g., "joint_pain") to human-readable format ("Joint Pain")
    human_readable_symptoms = [s.replace('_', ' ').title() for s in predictor_module.FULL_SYMPTOM_LIST]
    
    return jsonify({'symptoms': human_readable_symptoms})


# --- API Route for Disease Prediction ---
@app.route('/predict', methods=['POST'])
def handle_prediction():
    """Receives selected symptoms and returns a disease prediction and confidences."""
    
    # 1. Get data from the web request (JSON payload)
    data = request.get_json(silent=True)
    if not data or 'symptoms' not in data:
        return jsonify({'error': 'Invalid input: symptoms list missing in request body.'}), 400
    
    selected_symptoms = data['symptoms']
    
    print(f"DEBUG: Received symptoms: {selected_symptoms}")
    
    if not selected_symptoms:
        return jsonify({'prediction': 'No symptoms selected.', 'results': []}), 200

    try:
        # 2. Run the prediction
        primary_disease, top_results = predictor_module.predict_disease(selected_symptoms)

        print(f"DEBUG: Final prediction - {primary_disease}")

        # 3. Return the prediction result as a JSON response
        return jsonify({
            'prediction': primary_disease,
            'results': top_results,
            'status': 'success'
        })

    except Exception as e:
        print(f"Prediction failed: {e}")
        return jsonify({'error': 'An internal server error occurred during prediction.'}), 500


# --- Default Route for serving the HTML file ---
@app.route('/')
def index():
    return render_template('index.html')


# --- Health check route ---
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor_module.model is not None,
        'symptoms_count': len(predictor_module.FULL_SYMPTOM_LIST) if predictor_module.FULL_SYMPTOM_LIST else 0,
        'diseases_count': len(predictor_module.label_encoder.classes_) if predictor_module.label_encoder else 0
    })


# --- Server Run Command ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)