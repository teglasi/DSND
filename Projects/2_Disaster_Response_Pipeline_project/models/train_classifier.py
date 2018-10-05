# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.corpus import stopwords
import pandas as pd
import sqlalchemy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle
import re

def load_data(database_filepath):
    """
    Load the data set from the database.

    :param database_filepath: Path and filename to the database.
    :return: X: dataframe with features, Y: dataframe with labels, cat_names: category names
    """
    engine = sqlalchemy.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    df.drop(['original', 'genre'], axis=1, inplace=True)
    df.dropna(inplace=True)
    #Next drop the rows, which contain labels other than '0' or '1'.
    keep_labels = ['0', '1']
    df = df[df['related'].isin(keep_labels)]
    X = df.message.values
    Y = df.loc[:, 'related':'direct_report'].values
    cat_names = df.loc[:, 'related':'direct_report'].columns
    return X, Y, cat_names

def tokenize(text):
    """
    Tokenization function to process text data.

    :param text: Input text.
    :return: set of cleaned tokens
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def gridsearch(pipeline, X, Y):
    """
    Do GridSearchCV on the input model/pipeline.

    :param pipeline: Pipeline with basic parameters.
    :return: dictionary, the best parameters for the model
    """
    parameters = {
        'clf__estimator__learning_rate': [0.5, 0.7],
        'clf__estimator__n_estimators': [100, 200]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)
    cv.fit(X, Y)
    return cv.best_params_

def build_model(X, Y):
    """

    :return:
    """
    stop_w = set(stopwords.words('english'))
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, lowercase=True, stop_words=stop_w)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(XGBClassifier()))
    ])
    pipeline_parameters = {
        'features__text_pipeline__vect__ngram_range': (1, 2),
        'features__text_pipeline__vect__max_df': 0.75,
        'features__text_pipeline__vect__max_features': 5000,
        'features__transformer_weights': {'text_pipeline': 0.5, 'starting_verb': 1},
        'clf__estimator__n_estimators': 200,
        'clf__estimator__learning_rate': 0.5,
        'clf__estimator__scale_pos_weight': 2,
        'clf__estimator__max_delta_step': 3
    }
    pipeline.set_params(**pipeline_parameters)
    new_parameters = gridsearch(pipeline, X, Y)
    pipeline.set_params(**new_parameters)
    # print(pipeline.get_params(deep=False))
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.astype(int), y_pred.astype(int), target_names=category_names))

def save_model(model, model_filepath):
    """
    Save the model

    :param model: model to save
    :param model_filepath: Path and file name to destination.
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(X_train, Y_train)

        # print('Optimizing parameters with GridSearchCV...')

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()