from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictPipeline
from src.exception import CustomException
from src.logger import logging
import mlflow

application = Flask(__name__)
app = application


@app.route('/')
def home_page():
    try:
        return render_template('index.html')
    except Exception as e:
        raise CustomException(e)


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('form.html')
        else:
            data = CustomData(
                LIMIT_BAL=float(request.form.get('LIMIT_BAL')),
                AGE=float(request.form.get('AGE')),
                BILL_AMT1=float(request.form.get('BILL_AMT1')),
                BILL_AMT2=float(request.form.get('BILL_AMT2')),
                BILL_AMT3=float(request.form.get('BILL_AMT3')),
                BILL_AMT4=float(request.form.get('BILL_AMT4')),
                BILL_AMT5=float(request.form.get('BILL_AMT5')),
                BILL_AMT6=float(request.form.get('BILL_AMT6')),
                PAY_AMT1=float(request.form.get('PAY_AMT1')),
                PAY_AMT2=float(request.form.get('PAY_AMT2')),
                PAY_AMT3=float(request.form.get('PAY_AMT3')),
                PAY_AMT4=float(request.form.get('PAY_AMT4')),
                PAY_AMT5=float(request.form.get('PAY_AMT5')),
                PAY_AMT6=float(request.form.get('PAY_AMT6')),
                SEX=request.form.get('SEX'),
                EDUCATION=request.form.get('EDUCATION'),
                MARRIAGE=request.form.get('MARRIAGE'),
                PAY_1=request.form.get('PAY_1'),
                PAY_2=request.form.get('PAY_2'),
                PAY_3=request.form.get('PAY_3'),
                PAY_4=request.form.get('PAY_4'),
                PAY_5=request.form.get('PAY_5'),
                PAY_6=request.form.get('PAY_6')
            )
            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)
            results = pred[0]

            with mlflow.start_run():
                mlflow.log_param("input data", final_new_data)
                mlflow.log_param("results", str(results))

            return render_template('result.html', final_result=results)
    except Exception as e:
        logging.error('Exception occurred while running Flask API: {}'.format(e))
        #raise CustomException(e)
        raise CustomException("An error occurred", error_detail=str(e))



if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
