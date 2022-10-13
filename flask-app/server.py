import os
import sys
import json
import traceback

from flask import Flask
from flask import request
from waitress import serve

from classifier import TextClassifier, FitConfig, PredictConfig
# Using single processor:             python3 server.py 0.0.0.0 1234
# Using Gunicorn parallel processors: gunicorn --bind 0.0.0.0:1234 --workers 2 'server:app'

# Results: only Flask - 51s, Flask + waitress: 16s, Flask + waitress + Gunicorn: 6.5s
app = Flask(__name__)

@app.route('/fit', methods=['POST'])
def fit():
    try:
        texts = request.json['texts']
        labels = request.json['labels']

        config_dict = request.json['config']
        config = FitConfig(model_path=config_dict['model_path'], feature_type=config_dict['feature_type'])

        TextClassifier.fit(texts=texts, labels=labels, config=config)

        return {'success': True}

    except Exception as error:
        return {
            'success': False,
            'message': str(error),
            'traceback': traceback.format_exc(),
        }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        texts = request.json['texts']
        config_dict = request.json['config']
        config = PredictConfig(model_path=config_dict['model_path'], top_n=config_dict['top_n'])

        predicted = TextClassifier.predict(texts=texts, config=config)

        return {
            'success': True,
            'predicted': predicted,
        }
    except Exception as error:
        return {
            'success': False,
            'message': str(error),
            'traceback': traceback.format_exc(),
        }

# Only Flask
# if __name__ == '__main__':
#     if len(sys.argv) != 3:
#         print('Run `python server.py <HOST> <PORT>`')
#         sys.exit(1)

#     host = sys.argv[1]
#     port = sys.argv[2]

#     print(f'Start server on {host}:{port}')
#     app.run(host=host, port=port)

# Flask + waitress
if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Run `python server.py <HOST> <PORT>`')
        sys.exit(1)

    host = sys.argv[1]
    port = sys.argv[2]

    print(f'Start server on {host}:{port}')
    serve(app, host=host, port=port)