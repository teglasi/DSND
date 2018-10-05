### Table of Contents

1. [Project Summary](#summary)
2. [Project Details](#details)
3. [File Descriptions](#files)
4. [Licensing, Authors, Acknowledgements](#licensing)

The README file includes a summary of the project, how to run the Python scripts and web app, and an explanation of the files in the repository. Comments are used effectively and each function has a docstring.

## Project Summary<a name="summary"></a>

The 

## Project Details<a name="details"></a>

The

## File Descriptions<a name="files"></a>

The file structure:

- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md


## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data I used in this analysis was provided by Stack Overflow: Available [here](https://insights.stackoverflow.com/survey).