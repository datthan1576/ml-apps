import requests
model_path = 'test-model'

result_fit = requests.post(url='http://0.0.0.0:1234/fit',
                           json={
                               'texts': ['i love cats', 'cats are the best', 'what about dogs', 'i love dogs'],
                               'labels': ['cats', 'cats', 'dogs', 'dogs'],
                               'config': {'model_path': model_path, 'feature_type': 'bow'},
                               })
print(result_fit.json())