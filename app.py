
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model = pickle.load(open('models/model.pkl', 'rb'))

app = Flask(__name__)

# Define the categorical features and their possible values
categorical_columns = ['Company', 'TypeName', 'Weight_Category', 'Cpu_brand', 'Gpu_brand', 'Os']
feature_order = ['Company', 'TypeName', 'Ram', 'Weight_Category', 'Cpu_brand', 'HDD', 'SSD', 'Gpu_brand', 'Os']

def preprocess_input(user_input):
    """
    Preprocess form data to match the model's expected input.
    """
    # Convert form data to a DataFrame
    df = pd.DataFrame([user_input])
    
    # Apply one-hot encoding
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)
    
    # Ensure all expected columns are present, filling missing ones with 0
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0

    # Align the columns with the model's expected feature order
    df = df[model.feature_names_in_]

    return df.astype(float)

@app.route('/')
def home():
    return render_template('index_1.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        user_input = {
            "Company": request.form.get("Company"),
            "TypeName": request.form.get("TypeName"),
            "Ram": int(request.form.get("Ram")),
            "Weight_Category": request.form.get("Weight_Category"),
            "Cpu_brand": request.form.get("Cpu_brand"),
            "HDD": int(request.form.get("HDD")),
            "SSD": int(request.form.get("SSD")),
            "Gpu_brand": request.form.get("Gpu_brand"),
            "Os": request.form.get("Os"),
        }

        # Preprocess the input
        processed_data = preprocess_input(user_input)

        # Make prediction
        prediction = model.predict(processed_data)[0]

        # Return prediction to the template
        return render_template('index_1.html', prediction_text=f'Predicted Price: {round(prediction, 2)}')
    
    except Exception as e:
        # Log and handle the error
        print(f"Error: {e}")
        return render_template('index_1.html', prediction_text="An error occurred during prediction. Please try again.")

if __name__ == "__main__":
    app.run(debug=True)