# Credit Default Prediction

## Problem Statement:
Financial threats are becoming a significant concern for commercial banks due to the ever-improving financial industry. One of the 
most prominent threats faced by commercial banks is predicting the credit risk of their clients. The objective of this project is to
develop a predictive model that can estimate the probability of credit default based on the characteristics of credit card owners and 
their payment history.

Dataset: <https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset>

## Project Stages

### 1. Data Exploration:
  - Understand the dataset by analyzing its structure, patterns, and potential issues.
  - Identify anomalies and missing data that might affect model performance.

### 2. Data Cleaning:
  - Preprocess the dataset to handle missing values, outliers, and data inconsistencies.
  - Ensure the data is in the right format for modeling.

### 3. Feature Engineering:
  - Select, transform, or create features that enhance credit default prediction.
  - Use domain knowledge and data analysis to extract valuable information.

### 4. Model Building:
  - Experiment with various machine learning algorithms, such as logistic regression, random forests, and support vector machines.
  - Identify the most suitable model for credit default prediction.

### 5. Model Testing:
  - Evaluate model performance using relevant metrics to ensure accurate predictions.

### 6. Deployment:
  - Deploy the chosen model to a production environment, leveraging cloud platforms like AWS Elastic Beanstalk or EC2.
  - Create an API for users to input data and receive credit default predictions.

## Requirements

Before you begin, make sure you have the following requirements:

- Python 3.8 or later installed in your computer <https://www.python.org/downloads/>

## Setup and Usage

    1. **Clone the repository**:

        git clone <https://github.com/MKGourab/Credit_Card_Default_Prediction>

        cd your-repo-name

    2. Create a virtual Environment:

        conda create -p "environment name" python=3.8 -y

    3. Activate Virtual Environment:

        conda activate <Virtual Environment Path>

    4. Install the required packages:

        pip install -r requirements.txt

    5. Run the Flask application:

        python application.py

    6. Access the application:

        - Open your web browser and go to <http://localhost:5000> to interact with the app.

        - Fill in the required fields in the classification form. Enter the data inputs according to the descriptions and submit the form.

        - After submitting the form, you'll be directed to a page displaying the classification results. The predicted obesity classification will be shown.

## Results:
The goal of this project is to develop a solution that accurately predicts the probability of credit default based on credit card owner's
characteristics and payment history. The success of the project will be determined by the model's performance on the test data, where it
should demonstrate a strong ability to classify instances as credit default or non-default with high precision and recall.

The final output of the project will be a trained machine learning model, along with the API, that can be used to predict credit default
probabilities for new credit card applications or existing credit card holders.

## Project Structure

<pre>
<code>
project_root/
│
├── .ebextensions/
│   └── python.config            # Configuration for AWS Elastic Beanstalk deployment
│
├── .elasticbeanstalk/
│   └── config.yml              # Elastic Beanstalk configuration settings
│
├── Notebook/
│   ├── eda.ipynb                # Jupyter Notebook for Exploratory Data Analysis
│   ├── model_training.ipynb     # Jupyter Notebook for Model Training
│   └── data.csv                 # Raw data for analysis and training
│
├── templates/                   # Template files for web deployment 
│   └── template.html 
│
├── artifacts/
│   ├── model.pkl                # Pickled trained machine learning model
│   ├── preprocessor.pkl         # Pickled data preprocessor or feature transformer
│   ├── raw.csv                  # Raw data file (backup or reference)
│   ├── train.csv                # Processed training data
│   └── test.csv                 # Processed testing data 
│
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py           # Module for ingesting data from various sources
│   │   ├── data_transformation.py      # Module for data preprocessing and transformation
│   │   └── model_trainer.py            # Module for training machine learning models
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── prediction_pipeline.py  # Module for making predictions using trained models
│   │   └── training_pipeline.py    # Module for orchestrating the training process
│   │
│   ├── __init__.py
│   ├── exception.py            # Custom exception classes if needed
│   ├── logger.py               # Logging configuration and utility functions
│   └── utils.py                # General utility functions
│
├── application.py              # Flask application script
└── requirements.txt            # List of Python dependencies

</code>
</pre>


## Output is as shown below:

### Homepage:

![Screenshot_3](https://github.com/MKGourab/test/assets/104300031/8a496d3b-ea21-48ea-b31a-d1edac56ea54)

### After_User_Input:

![Screenshot_4](https://github.com/MKGourab/test/assets/104300031/ebf1d7f7-d1e6-4df7-8c76-db084a8e70b4)

### Result:

![Screenshot_5](https://github.com/MKGourab/test/assets/104300031/991436f1-ff1e-4773-bbac-23ef8e58e927)
