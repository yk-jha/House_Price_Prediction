from flask import Flask , request , render_template # type: ignore
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline


applicaton = Flask(__name__)

app = applicaton
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            location=request.form.get('location'),
            size=request.form.get('size'),
            total_sqft=float(request.form.get('total_sqft')),
            price_per_sqft=float(request.form.get('price_per_sqft')),
            bhk=request.form.get('bhk'),
            bath=float(request.form.get('bath'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.Predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(port=5001 , host="0.0.0.0" )