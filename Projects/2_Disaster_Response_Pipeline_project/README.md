### Table of Contents

1. [Project Summary](#summary)
2. [Project Details](#details)
3. [File Descriptions](#files)
4. [Project Notes](#notes)
5. [Licensing, Authors, Acknowledgements](#licensing)

## Project Summary<a name="summary"></a>

Disasters are followed by millions of communications in either direct or indirect ways.
Disaster response organizations must filter the messages based on their importance and relevancy and dispatch them to the apripriate body.
This project provides a data pipeline to prepare messages, a machine learning pipeline to categorize messages and a web-based application to classify new messages.
 
## Project Details<a name="details"></a>

The project consists of three parts:
###ETL Pipeline
The first part is the Extract, Transform, and Load process.

**process_data.py** loads and merges data from the original data files: disaster_categories.csv and disaster_messages.csv.
Duplicated data is dropped and the data set is saved into an SQL database: **DisasterResponse.db**
   
###Machine Learning Pipeline
The second part provides a Machine Learning pipeline with a supervised learning model.
The model is based on sklearn's MultiOutputClassifier, which leverages the XGBClassifier.

**train_classifier.py** loads the data from the SQL database, splits it into training and testing sets, and builds, trains, evaluates and saves the classifier model.
Building the model is done through GridSearch Cross-Validation, which is a technique to optimize the learning algorithm's parameters.

The current script contains the following parameter-grid:

```    
parameters = {
        'clf__estimator__learning_rate': [0.5, 0.7],
        'clf__estimator__n_estimators': [100, 200]}
```

The original parameter-grid was more extensive but avioded in the current code due to the long time it takes to execute.
```
    original_parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1),(1, 2)),
        'features__text_pipeline__vect__max_df': (0.5, 0.75, 1),
        'features__text_pipeline__vect__max_features': (None, 5000, 10000),
        'clf__estimator__n_estimators': [100, 200],
        'clf__estimator__learning_rate': [0.3, 0.5, 1.0],
        'clf__estimator__scale_pos_weight': [2, 3],
        'clf__estimator__max_delta_step': [2, 3, 5]
    }
```

###Web APP
The Flask-based web application provides an interface to interact with the model trained in the previous part.
The application takes a text message as input and returns classification results for all categories. 

###Running the Project

Running the project takes three steps:

###Step 1:
Run **process_data.py** from the _data_ directory:

    python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

    The arguments are:
    1.  input data file 1 (messages data)
    2.  input data file 2 (categories data)
    3.  database name for the output data
This scipt creates the database file, which is used as the input for the second step.

###Step 2:
Run **train_classifier.py** from the _model_ directory:

    python train_classifier.py ../data/DisasterResponse.db classifier.pkl
    
    The arguments are:
    1.  input database path+file_name
    2.  serialized classifier model (this will be used by the web app for classification)

This script cleans and prepares text data, optimizes the model via GridSearchCV and saves the model as a pickle file.

###Step 3:
Run **run.py** from the _app_ directory:

    python run.py
    
This script runs a flask web application on the local host. The application is available through a browser at http://localhost:3001.

## File Descriptions<a name="files"></a>

Project files are organized into the following file structure:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- DisasterResponse.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```
**DisasterResponse.db** and **classifier.pkl** and not part of the GitHub repository. Those files are generated by running the corresponding python scripts.

## Project Notes<a name="notes"></a>
The original dataset is imbalanced. Due to the very low number of some labels in the data set, precision and recall is negatively impacted.
This effect is mitigated by tuning the **scale_pos_weight** and **max_delta_step** parameters of the classifier.
**Scale_pos_weight** gives more wights to samples based on the ration of positive and negative labels in the class.
**max_delta_step** makes the update step more conservative during training.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data, used in this analysis, was provided by [Figure Eight](https://www.figure-eight.com/).