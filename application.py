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
        error_message = f'Error rendering index.html: {str(e)}'
        logging.error(error_message)
        raise CustomException(error_message)

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('test.html')
        else:
            form_data = request.form
            data = CustomData(**form_data.to_dict())
            final_new_data = data.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            pred = predict_pipeline.predict(final_new_data)
            results = pred[0]

            # Log input data and results using MLflow (uncomment if needed)
            # with mlflow.start_run():
            #     mlflow.log_params(form_data.to_dict())
            #     mlflow.log_params({'results': str(results)})

            return render_template('test.html', final_result=results)
    except Exception as e:
        error_message = f'Exception occurred while running Flask API: {str(e)}'
        logging.error(error_message)
        raise CustomException(error_message)

if __name__ == "__main__":
    app.run(debug=True)