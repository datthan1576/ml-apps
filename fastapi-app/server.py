import sys
import inspect
import traceback

import uvicorn

from fastapi import FastAPI
from pydantic import create_model

from classifier import TextClassifier
from structures import FitConfig, PredictConfig, Texts, ReturnValue, PredictReturnValue

# python3 server.py 0.0.0.0 1234
# gunicorn --bind 0.0.0.0:1234 -w 2 -k uvicorn.workers.UvicornWorker server:app
# result-200 requests: FastAPT + Unicorn : 11.9s, FastAPI + Unicorn + Gunicorn: 8.36s

app = FastAPI()

def get_params(method):
    return {k: (v.annotation, ...) for k, v in inspect.signature(method).parameters.items()}

@app.post("/fit", response_model=ReturnValue, name='Fit')
async def fit(request: create_model('FitInput', **get_params(TextClassifier.fit))):
    try:
        TextClassifier.fit(texts=request.texts, labels=request.labels, config=request.config)

        return ReturnValue(success=True)

    except Exception as error:
        return ReturnValue(
            success=False,
            message=str(error),
            traceback=str(traceback.format_exc()),
        )

@app.post("/predict", response_model=PredictReturnValue, name='Predict')
async def predict(request: create_model('PredictInput', **get_params(TextClassifier.predict))):
    try:
        return PredictReturnValue(success=True,
                                  predicted=TextClassifier.predict(texts=request.texts, config=request.config))
    except Exception as error:
        return PredictReturnValue(
            success=False,
            message=str(error),
            traceback=str(traceback.format_exc()),
        )

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Run `python server.py <HOST> <PORT>`')
        sys.exit(1)

    host = sys.argv[1]
    port = int(sys.argv[2])

    uvicorn.run('server:app', host=host, port=port)