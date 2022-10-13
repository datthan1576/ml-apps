import requests
from sklearn.datasets import fetch_20newsgroups

model_path = 'model-test'

# requests.post(url='http://0.0.0.0:1234/fit',
#               json={
#                   'texts': {
#                       'values': ['i love cats', 'cats are the best',
#                                  'what about dogs', 'i love dogs'],
#                   },
#                   'labels': {
#                       'values': ['cats', 'cats', 'dogs', 'dogs'],
#                   },
#                   'config': {
#                       'model_path': model_path,
#                       'feature_type': 'bow',
#                   },
#               })

# requests.post(url='http://0.0.0.0:1234/fit',
#               json={
#                   'texts': {
#                       'values': ['i like dogs', 'i walk with cats'],
#                   },
#                   'config': {
#                       'model_path': model_path,
#                       'top_n': 2,
#                   },
#               })

# dataset = fetch_20newsgroups()
# texts = dataset['data']
# labels = [dataset['target_names'][i] for i in dataset['target']]

# result_fit = requests.post(url='http://0.0.0.0:1234/fit',
#                            json={
#                                'texts': {'values': texts},
#                                'labels': {'values': labels},
#                                'config': {'model_path': model_path, 'feature_type': 'bow'},
#                                })
# print(result_fit.json())


import time
import asyncio
import aiohttp

from sklearn.datasets import fetch_20newsgroups

num_requests = 200
model_path = 'test-model'

async def post(session):
    async with session.post(url='http://0.0.0.0:1234/predict',
                            json={
                                'texts': {'values': texts[: 5000]},
                                'config': {'model_path': model_path, 'top_n': 3}
                            }) as response:
        await response.read()

async def main():
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[post(session) for _ in range(num_requests)])

dataset = fetch_20newsgroups()
texts = dataset['data']
labels = [dataset['target_names'][i] for i in dataset['target']]

try:
    ts = time.time()
    asyncio.run(main())
    print(f'Elapsed time for {num_requests} requests: {round(time.time() - ts, 2)} sec.')
except:
    pass