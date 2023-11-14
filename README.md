# Starbucks-Udacity

## Project Description
This project is the final project for Udacity's Data Science Nanodegree program.

The goal of the project was to create an application that predicts whether a user will respond to an offer received in Starbucks mobile app.

The application receives several datasets and outputs a csv file with predictions.

Simulated data was provided, which mimics customer behavior on the Starbucks rewards mobile app.

## Data
The description below was provided at Udacity.

The data is contained in three files:

* portfolio.json - containing offer ids and meta data about each offer (duration, type, etc.)
* profile.json - demographic data for each customer
* transcript.json - records for transactions, offers received, offers viewed, and offers completed

Here is the schema and explanation of each variable in the files:

**portfolio.json**
* id (string) - offer id
* offer_type (string) - type of offer ie BOGO, discount, informational
* difficulty (int) - minimum required spend to complete an offer
* reward (int) - reward given for completing an offer
* duration (int) - time for offer to be open, in days
* channels (list of strings)

**profile.json**
* age (int) - age of the customer 
* became_member_on (int) - date when customer created an app account
* gender (str) - gender of the customer (note some entries contain 'O' for other rather than M or F)
* id (str) - customer id
* income (float) - customer's income

**transcript.json**
* event (str) - record description (ie transaction, offer received, offer viewed, etc.)
* person (str) - customer id
* time (int) - time in hours since start of test. The data begins at time t=0
* value - (dict of strings) - either an offer id or transaction amount depending on the record

Apart from the files provided by Udacity, there are several files present in the repository:
* Starbucks_Capstone_notebook.ipynb - Jupyter notebook containing end-to-end project
* classifier.py - a file containing the classifier
* classifier_functions.py - a file containing all the functions the classifier uses
* analysis_functions.py - a file containing all functions used during model preparation
* xtest.csv, ytest.csv - test datasets
* preds.csv - csv output by the classifier

## Requirements

All you need is python 3 and Jupyter notebooks. All the libraries are imported in the files.

## Usage

I'd recommend starting with ipynb file to understand the data and the calssifier. You can simply open it and run all cells.

Then, if you want to use the classifier

1. In classifier.py comment out a single line under # Uncomment 
2. In cmd print ipython to enable python functionality
3. Import Classifier() class from classifier.py ( from classifier.py import Classifier )
4. Instantiate a classifier object ( clf = Classifier() )
5. Fit classifier inputing paths to portfolio, profile, 

