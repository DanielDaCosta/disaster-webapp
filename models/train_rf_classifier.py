import sys
import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///data/{database_filepath}')
    df = pd.read_sql_table('messages', engine)

    output_columns = [
        'related', 'request', 'offer', 'aid_related',
        'medical_help', 'medical_products', 'search_and_rescue', 'security',
        'military', 'child_alone', 'water', 'food', 'shelter', 'clothing',
        'money', 'missing_people', 'refugees', 'death', 'other_aid',
        'infrastructure_related', 'transport', 'buildings', 'electricity',
        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
        'other_weather', 'direct_report'
        ]
    X = df['message']
    y = df[output_columns]
    return X, y, output_columns


def tokenize(text):
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # '@' mention. Even tough @ adds some information to the message,
    # this information doesn't add value build the classifcation model
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)

    # Dealing with URL links
    url_regex = '''http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|
                   (?:%[0-9a-fA-F][0-9a-fA-F]))+'''
    text = re.sub(url_regex, 'urlplaceholder', text)
    # A lot of url are write as follows: http bit.ly. Apply Regex for these
    # cases
    utl_regex_2 = 'http [a-zA-Z]+\.[a-zA-Z]+'
    text = re.sub(utl_regex_2, 'urlplaceholder', text)
    # Other formats: http : //t.co/ihW64e8Z
    utl_regex_3 = 'http \: //[a-zA-Z]\.(co|com|pt|ly)/[A-Za-z0-9_]+'
    text = re.sub(utl_regex_3, 'urlplaceholder', text)

    # Hashtags can provide useful informations. Removing only ``#``
    text = re.sub('#', ' ', text)

    text = text.lower()
    # Ponctuation Removal
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [tok for tok in tokens if tok not in stop_words]
    return tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(
            bootstrap=True, n_estimators=150, max_features='sqrt',
            max_depth=8, max_samples=0.9, min_samples_split=10,
            class_weight='balanced', random_state=42)))
    ])
    parameters = {
        'clf__estimator__min_samples_split': [8, 9, 10]
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    f1_score_results = []
    for col_idx, col in enumerate(category_names):
        print(f'{col} accuracy \n')
        f1_score_results.append(f1_score(y_test[col], y_pred[:, col_idx],
                                         average='macro'))
        print(classification_report(y_test[col], y_pred[:, col_idx]))
    # Creating a general metric that combines all 36 classes
    print('Total :', np.sum(f1_score_results))


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test =\
            train_test_split(X, Y, test_size=0.2, random_state=42)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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