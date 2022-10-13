import requests
from sklearn.datasets import fetch_20newsgroups

model_path = 'test-model'

dataset = fetch_20newsgroups()
texts = dataset['data']
labels = [dataset['target_names'][i] for i in dataset['target']]

result_fit = requests.post(url='http://0.0.0.0:1234/fit',
                           json={
                               'texts': texts,
                               'labels': labels,
                               'config': {'model_path': model_path, 'feature_type': 'bow'},
                               })
print(result_fit.json())