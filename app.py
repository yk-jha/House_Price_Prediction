from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        # Load preprocessor to get location and size categories
        try:
            with open('artifacts/preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            encoder = preprocessor.named_transformers_['cat_pipelines'].named_steps['one_hot_encoder']
            locations = encoder.categories_[0].tolist()  # Location categories
            sizes = encoder.categories_[1].tolist()      # Size categories
        except Exception as e:
            return render_template('home.html', error=f"Error loading categories: {str(e)}")
        
        return render_template('home.html', locations=locations, sizes=sizes)
    
    else:
        # Handle POST request for prediction
        try:
            data = CustomData(
                location=request.form.get('location'),
                size=request.form.get('size'),
                total_sqft=float(request.form.get('total_sqft')),
                price_per_sqft=float(request.form.get('price_per_sqft')),
                bhk=float(request.form.get('bhk')),
                bath=float(request.form.get('bath'))
            )
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.Predict(pred_df)
            print("After Prediction")

            # Reload categories to pass back to the template
            with open('artifacts/preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            encoder = preprocessor.named_transformers_['cat_pipelines'].named_steps['one_hot_encoder']
            locations = encoder.categories_[0].tolist()
            sizes = encoder.categories_[1].tolist()

            return render_template('home.html', results=results[0], locations=locations, sizes=sizes)
        except Exception as e:
            # Reload categories for error case
            with open('artifacts/preprocessor.pkl', 'rb') as f:
                preprocessor = pickle.load(f)
            encoder = preprocessor.named_transformers_['cat_pipelines'].named_steps['one_hot_encoder']
            locations = encoder.categories_[0].tolist()
            sizes = encoder.categories_[1].tolist()
            return render_template('home.html', error=f"Prediction failed: {str(e)}", locations=locations, sizes=sizes)

if __name__ == "__main__":
    app.run(port=5001, host="0.0.0.0")