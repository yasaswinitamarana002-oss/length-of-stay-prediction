from flask import Flask, render_template, request
import joblib
import pandas as pd
app = Flask(__name__)

# Load model safely
try:
    model = joblib.load("los_prediction_model.pkl")
except FileNotFoundError:
    model = None
    print("Error: Model file 'los_prediction_model.pkl' not found. Make sure it exists in the project folder.")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template("index.html", prediction_text="Model not loaded. Cannot make predictions.")
    
    try:
        # Get inputs and validate
        age = request.form.get('age')
        satisfaction = request.form.get('satisfaction')
        service = request.form.get('service')

        if not age or not satisfaction or not service:
            return render_template("index.html", prediction_text="Please fill out all fields.")

        age = int(age)
        satisfaction = int(satisfaction)

        # Create DataFrame with the same columns as the model expects
        new_patient = pd.DataFrame({
            'age': [age],
            'service': [service],
            'satisfaction': [satisfaction]
        })

        # Make prediction
        prediction = model.predict(new_patient)
        pred_days = round(prediction[0], 2)

        return render_template("index.html",
                               prediction_text=f"Predicted Length of Stay: {pred_days} days")

    except ValueError:
        return render_template("index.html", prediction_text="Please enter valid numbers for age and satisfaction.")
    except Exception as e:
        # Catch-all for any other unexpected errors
        return render_template("index.html", prediction_text=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)