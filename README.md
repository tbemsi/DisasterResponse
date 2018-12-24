# Disaster Response Pipeline Project

## Project Overview

In this project, I will be creating a machine learning pipeline to categorize messages sent during disaster events.

The project will include a web app where an emergency worker can input a new message and get classification results
in several categories. The web app will also display visualizations of the data.

## Files and folders in this repository

There are three folders in this repository:
* __data__: Contains the csv files with the messages and message categories, as well as the file `process_data.py`, which
contains code for loading and cleaning the data.
* __models__: Contains the file `train_classifier.py`, which contains code for creating a machine learning pipeline on the
clean data, training it, and then saving it as a pickle file.
* __app__: Contains html templates for the Flask web app, as well as `run.py`, which contains code for the web application.


## Repository Layout
The following is the layout of this repository
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
|- DisasterResponse.db   # database where clean data is saved

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```

## How to run the code
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Libraries Used
* Flask(1.0.2)
* nltk(3.4)
* numpy(1.15.3)
* pandas(0.23.4)
* plotly(3.3.0)
* scikit-learn(0.20.0)
* SQLAlchemy(1.2.14)

## Authors and Acknowledgements
Thanks to Udacity and FigureEight for providing the data for this project.
