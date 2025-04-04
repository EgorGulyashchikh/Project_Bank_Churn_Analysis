from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

CORS(app)

# Download model and data
def load_model():
    if not os.path.exists('churn_model.pkl') or not os.path.exists('train_data.pkl'):
        raise FileNotFoundError("Model files not found. Please run model_training.py first")
    
    model = joblib.load('churn_model.pkl')
    X_train = joblib.load('train_data.pkl')
    explainer = shap.TreeExplainer(model, X_train)
    
    return model, explainer, X_train.columns

try:
    model, explainer, model_columns = load_model()
except Exception as e:
    print(f"Error loading model: {e}")
    model, explainer, model_columns = None, None, None

# Web-interface
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# API for predictions
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        # Create DataFrame
        input_df = pd.DataFrame([{
            'CreditScore': int(data['CreditScore']),
            'Age': int(data['Age']),
            'Balance': float(data['Balance']),
            'EstimatedSalary': float(data['EstimatedSalary']),
            'Geography_Germany': 1 if data['Geography'] == 'Germany' else 0,
            'Geography_Spain': 1 if data['Geography'] == 'Spain' else 0,
            'Gender_Male': 1 if data['Gender'] == 'Male' else 0,
            'Tenure_1': 1 if data["Tenure"] == 1 else 0,
            'Tenure_2': 1 if data["Tenure"] == 2 else 0,
            'Tenure_3': 1 if data["Tenure"] == 3 else 0,
            'Tenure_4': 1 if data["Tenure"] == 4 else 0,
            'Tenure_5': 1 if data["Tenure"] == 5 else 0,
            'Tenure_6': 1 if data["Tenure"] == 6 else 0,
            'Tenure_7': 1 if data["Tenure"] == 7 else 0,
            'Tenure_8': 1 if data["Tenure"] == 8 else 0,
            'Tenure_9': 1 if data["Tenure"] == 9 else 0,
            'Tenure_10': 1 if data["Tenure"] == 10 else 0,
            'NumOfProducts_2': 1 if data["NumOfProducts"] == 2 else 0,
            'NumOfProducts_3': 1 if data["NumOfProducts"] == 3 else 0,
            'NumOfProducts_4': 1 if data["NumOfProducts"] == 4 else 0,
            'HasCrCard_1': 1 if data["HasCrCard"] == 1 else 0,
            'IsActiveMember_1': 1 if data["IsActiveMember"] == 1 else 0
        }])
        
        # Make sure that order is right
        input_df = input_df[model_columns]
        
        # Prediction
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]
        
        try:
            shap_output = explainer(input_df, check_additivity=False)
            
            # Work with different output formats
            if hasattr(shap_output, 'values'):
                shap_values = shap_output.values
                base_value = shap_output.base_values
            else:
                shap_values = explainer.shap_values(input_df)
                base_value = explainer.expected_value
            
            # Normalization of the format for binary classification
            if isinstance(shap_values, list):
                shap_values = shap_values[1] 
            
            if isinstance(base_value, list):
                base_value = base_value[1]
            
            # Convert to numpy array, if necessary
            shap_values = np.array(shap_values)
            if len(shap_values.shape) == 1:
                shap_values = shap_values.reshape(1, -1)
            
            # Visualization
            plt.figure()
            shap.force_plot(
                base_value,
                shap_values[0],
                input_df.iloc[0],
                matplotlib=True,
                show=False,
                text_rotation=45,
            )
            plt.savefig('static/shap_force.png', bbox_inches='tight')
            plt.close()
            
            shap_available = True
        except Exception as shap_error:
            print(f"SHAP visualization failed: {shap_error}")
            shap_available = False
        
        # Forming a response
        result = {
            'prediction': int(prediction),
            'probability': float(proba),
            'shap_available': shap_available
        }
        
        if shap_available:
            result.update({
                'shap_summary': 'static/shap_summary.png',
                'shap_force': 'static/shap_force.png'
            })
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting server at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)