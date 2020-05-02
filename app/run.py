import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.graph_objs as go

app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


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
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
