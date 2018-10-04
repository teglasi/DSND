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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
import pickle
import re

def load_data(database_filepath):
    engine = sqlalchemy.create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine)
    df.drop(['original', 'genre'], axis=1, inplace=True)
    df.dropna(inplace=True)
    keep_labels = ['0', '1']
    df = df[df['related'].isin(keep_labels)]
    X = df.message.values
    Y = df.loc[:, 'related':'direct_report'].values
    return X, Y, df.loc[:, 'related':'direct_report'].columns

def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def build_model():
    stop_w = set(stopwords.words('english'))
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize, lowercase=True, stop_words=stop_w)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    pipeline_parameters = {
        'features__text_pipeline__vect__ngram_range': (1, 2),
        'features__text_pipeline__vect__max_df': 0.75,
        'features__text_pipeline__vect__max_features': 5000,
        'features__transformer_weights': {'text_pipeline': 0.5, 'starting_verb': 1},
        'clf__estimator__n_estimators': 150,
        'clf__estimator__learning_rate': 0.5
    }
    pipeline.set_params(**pipeline_parameters)
    # print(pipeline.get_params(deep=False))
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.astype(int), y_pred.astype(int)))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
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