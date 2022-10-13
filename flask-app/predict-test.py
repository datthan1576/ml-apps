import requests

model_path = 'test-model'
result_predict = requests.post(url='http://0.0.0.0:1234/predict',
                               json={
                                   'texts': ['i like dogs', 'i walk with cats'],
                                   'config': {'model_path': model_path, 'top_n': 2},
                               })
print(result_predict.json())