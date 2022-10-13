import time
import asyncio
import aiohttp

from sklearn.datasets import fetch_20newsgroups

num_requests = 20
model_path = 'test-model'
dataset = fetch_20newsgroups()
texts = dataset['data']
labels = [dataset['target_names'][i] for i in dataset['target']]

async def post(session):
    async with session.post(url='http://0.0.0.0:1234/predict',
                            json={
                                'texts': texts[: 5000],
                                'config': {'model_path': model_path, 'top_n': 3}
                            }) as response:
        await response.read()

async def main():
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*[post(session) for _ in range(num_requests)])

try:
    ts = time.time()
    asyncio.run(main())
    print(f'Elapsed time for {num_requests} requests: {round(time.time() - ts, 2)} sec.')
except:
    pass