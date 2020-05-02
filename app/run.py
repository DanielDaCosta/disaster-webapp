import json
import plotly
import pandas as pd

import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.graph_objs as go

app = Flask(__name__)


def tokenize(text):
    """ Tokenize string

    Args:
        text(string)
    Returns:
        tokens(list): tokens in list of strings format
    """
    text = text.lower()
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    # '@' mention. Even tough @ adds some information to the message,
    # this information doesn't add value build the classifcation model
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)

    # Dealing with URL links
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]'
                 '|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
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

    # Contractions
    text = re.sub(r"what's", 'what is ', text)
    text = re.sub(r"can't", 'cannot', text)
    text = re.sub(r"\'s", ' ', text)
    text = re.sub(r"\'ve", ' have ', text)
    text = re.sub(r"n't", ' not ', text)
    text = re.sub(r"im", 'i am ', text)
    text = re.sub(r"i'm", 'i am ', text)
    text = re.sub(r"\'re", ' are ', text)
    text = re.sub(r"\'d", ' would ', text)
    text = re.sub(r"\'ll", ' will ', text)

    # Operations and special words
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub('foof', 'food', text)
    text = re.sub('msg', 'message', text)
    text = re.sub(' u ', 'you', text)

    # Ponctuation Removal
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    tokens = [tok for tok in tokens if tok not in stop_words]
    return tokens


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    graphs = []
    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=genre_names,
            y=genre_counts
        )
    )
    figure.update_layout(
        go.Layout(
            title="Distribution of Message Genres",
            title_x=0.5,
            yaxis_title="Count",
            xaxis_title=f"Genre",
            plot_bgcolor="rgba(0,0,0,0)"
        )
    )
    graphs.append(dict(data=figure.data, layout=figure.layout))
    # graphs = [
    #     {
    #         'data': [
    #             Bar(
    #                 x=genre_names,
    #                 y=genre_counts
    #             )
    #         ],

    #         'layout': {
    #             'title': 'Distribution of Message Genres',
    #             'yaxis': {
    #                 'title': "Count"
    #             },
    #             'xaxis': {
    #                 'title': "Genre"
    #             }
    #         }
    #     }
    # ]

    # Plot Outputs Columns Distributions
    output_columns = [
        'related', 'request', 'offer', 'aid_related',
        'medical_help', 'medical_products', 'search_and_rescue', 'security',
        'military', 'child_alone', 'water', 'food', 'shelter', 'clothing',
        'money', 'missing_people', 'refugees', 'death', 'other_aid',
        'infrastructure_related', 'transport', 'buildings', 'electricity',
        'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure',
        'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold',
        'other_weather', 'direct_report']
    for i, col in enumerate(output_columns):
        counts = df.groupby(col).count()['id']
        total_rows = df.shape[0]
        names = col.replace('_', ' ').title()
        figure = go.Figure()

        figure.add_trace(
            go.Bar(
                x=counts.index,
                y=counts,
                text=round(counts[counts.index]/total_rows*100, 2)
                .apply(lambda x: str(x) + '%'),
                textposition='outside',
                cliponaxis=False
            )
        )

        figure.update_layout(
            go.Layout(
                title=f'{names}',
                title_x=0.5,
                yaxis_title="Count",
                xaxis_title=f"{names}",
                plot_bgcolor="rgba(0,0,0,0)"
            )
        )
        figure.update_traces(
            marker_color='rgb(158,202,225)',
            marker_line_color='rgb(8,48,107)',
            marker_line_width=1.5, opacity=0.6)
        graphs.append(dict(data=figure.data, layout=figure.layout))

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/classify')
def classify():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
