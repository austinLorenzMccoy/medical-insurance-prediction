from flask import Flask, request, render_template
import numpy as np
import pandas as pd

# Import prediction pipeline
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Collect form data
            data = CustomData(
                age=int(request.form.get('age')),
                sex=request.form.get('sex'),
                bmi=float(request.form.get('bmi')),
                children=int(request.form.get('children')),
                smoker=request.form.get('smoker'),
                region=request.form.get('region')
            )
            
            # Convert the data to a DataFrame
            pred_df = data.get_data_as_data_frame()
            
            # Predict using the pipeline
            predict_pipeline = PredictPipeline()
            result = predict_pipeline.predict(pred_df)[0]  # Assuming single prediction
            
            # Format result
            formatted_result = f"${result:,.2f}"  # Example: "$8.64"
            
            # Render with results
            return render_template('index.html', results=formatted_result)
        except Exception as e:
            return render_template('index.html', results=f"Error: {str(e)}")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
