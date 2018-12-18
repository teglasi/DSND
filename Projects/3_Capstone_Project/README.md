### Table of Contents

1. [Project Overview](#overview)
2. [Project Statement](#statement)
3. [File Descriptions](#files)
4. [Results of the Analysis](#results)
5. [Licensing, Authors, Acknowledgements](#licensing)

## Project Overview<a name="overview"></a>

This project provides a data analyzis on demographics data for customers of a mail-order sales company in Germany, comparing it against demographics information for the general population.
Based on this demographics information, a model is built to identify potential future customers from a third dataset.
 
## Project Statement<a name="statement"></a>

The project consists of three parts:

**Part 1: Customer Segmentation Report**

Using unsupervised learning techniques, I describe the relationship between the demographics of the company's existing customers and the general population of Germany. PCA is applied for dimensionality reduction and Kmeans for clustering. We expect to see the main differences between customers and general population by comparing the most and least represented clusters between the data sets.

**Part 2: Supervised Learning Model**

Based on the results of Part 1, I build a prediction model to decide, whether an individual of the general population is likely to become a customer of the mail-order company.

**Part 3: Kaggle Competition**

In the last part, the supervised model is used to make predictions on the campaign data as part of a Kaggle Competition.

## File Descriptions<a name="files"></a>


There are four data files, an IPython Notebook and a helper library associated with this project:

- Udacity_AZDIAS_052018.csv: Demographics data for the general population of Germany; 891 211 persons (rows) x 366 features (columns).
- Udacity_CUSTOMERS_052018.csv: Demographics data for customers of a mail-order company; 191 652 persons (rows) x 369 features (columns).
- Udacity_MAILOUT_052018_TRAIN.csv: Demographics data for individuals who were targets of a marketing campaign; 42 982 persons (rows) x 367 (columns).
- Udacity_MAILOUT_052018_TEST.csv: Demographics data for individuals who were targets of a marketing campaign; 42 833 persons (rows) x 366 (columns).
- Arvato.ipynb
- arvatoutils.py: Custom library created for this project

## Results of the Analysis<a name="results"></a>

The result of the analysis is described in the post available [here](https://medium.com).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

The data, used in this analysis, was provided by [Arvato Financial Services](https://www.arvato.com/).
