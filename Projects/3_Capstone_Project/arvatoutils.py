"""Helper file for Udacity Data Scientist Nanodegree Project"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


rnd_state = 16

def find_sparse_columns(dfs, min_missing_ratio):
    
    """Find columns in an array of dataframes, where the ratio
    of missing values of the column is higher then the min_missing_ratio parameter.
    
    Parameters
    ----------
    dfs : array of pandas.DataFrames
    min_missing_ratio : float, sets the ratio of missing values in a column
    
    Output
    ------
    List of column names, where the ratio of missing values is higher then min_missing_ratio.
    """
    
    limit = []
    cols_2_drop = []
    for i in dfs:
        limit = min_missing_ratio * i.shape[0]
        temp = i.isnull().sum().sort_values()
        cols_2_drop.extend(temp[temp > limit].index)
    return cols_2_drop

def remove_col_by_name(df, s):

    """Drop columns of a dataframe, if string s is part of the column name.

    Parameters
    ----------
    df : pandas.DataFrame
    s : str, string to look for in the column names

    Output
    ------
    List of column names, where the ratio of missing values is higher then min_missing_ratio.
    """

    return df.drop(list(df.filter(regex = s)), axis=1, inplace=True)

def create_cat_codes(df1, df2):

    """Create integer category codes for two dataframes based on label sets combined from both dataframes.

    Parameters
    ----------
    df1, df2 : pandas.DataFrames

    Output
    ------
    Two dataframes with coded categories.
    """

    combined = pd.concat([df1, df2])
    for col in combined.columns:
        combined[col] = combined[col].astype('category').cat.codes
    df1 = combined.iloc[:df1.shape[0], :]
    df2 = combined.iloc[df1.shape[0]:, :]
    return df1, df2

def feat_relevance(df):

    """Create integer category codes for two dataframes based on label sets combined from both dataframes.

    Parameters
    ----------
    df: pandas.DataFrames

    Output
    ------
    Two dataframes with coded categories.
    """

    score_df = pd.DataFrame(columns=['feature', 'score'])
    scoring = []
    for i in df.columns:
        temp_df = df.drop([i], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(temp_df, df[i], test_size=0.25, random_state=rnd_state)
        classifier = RandomForestClassifier(n_estimators=20, criterion='gini', random_state=rnd_state)
        classifier.fit(X_train, y_train)
        score = classifier.score(X_test, y_test)
        scoring.append([i, score])
    score_df = pd.DataFrame(scoring, columns=['feature', 'score'], index=None)
    return score_df

def pca_results(good_data, pca):
	'''
	Create a DataFrame of the PCA results
	Includes dimension feature weights and explained variance
	Visualizes the PCA results
	'''

	# Dimension indexing
	dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]

	# PCA components
	components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
	components.index = dimensions

	# PCA explained variance
	ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
	variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
	variance_ratios.index = dimensions

	# Create a bar plot visualization
	fig, ax = plt.subplots(figsize = (14,8))

	# Plot the feature weights as a function of the components
	components.plot(ax = ax, kind = 'bar');
	ax.set_ylabel("Feature Weights")
	ax.set_xticklabels(dimensions, rotation=0)


	# Display the explained variance ratios
	for i, ev in enumerate(pca.explained_variance_ratio_):
		ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n          %.4f"%(ev))

	# Return a concatenated DataFrame
	return pd.concat([variance_ratios, components], axis = 1)
