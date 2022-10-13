import pickle
import shutil
import numpy as np

from pathlib import Path
from typing import List, Literal, Union, Tuple
from dataclasses import dataclass
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression

@dataclass
class FitConfig:
    model_path: str
    feature_type: Union[Literal['tf-idf'], Literal['bow']]

@dataclass
class PredictConfig:
    model_path: str
    top_n: int

class TextClassifier:
    @staticmethod
    def fit(texts: List[str], labels: List[str], config: FitConfig) -> None:
        if config.feature_type == 'tf-idf':
            vectorizer = TfidfVectorizer()

        elif config.feature_type == 'bow':
            vectorizer = CountVectorizer()

        else:
            raise ValueError(f'Unknown config.feature_type: "{config.feature_type}"')

        data = vectorizer.fit_transform(texts)
        model = LogisticRegression()
        model.fit(data, labels)

        model_path = Path(config.model_path)

        if model_path.exists():
            shutil.rmtree(model_path)
        model_path.mkdir()

        with open(model_path / 'model.pkl', 'wb') as fout:
            pickle.dump(model, fout)

        with open(model_path / 'vectorizer.pkl', 'wb') as fout:
            pickle.dump(vectorizer, fout)

    @staticmethod
    def predict(texts: List[str], config: PredictConfig) -> List[List[Tuple[str, float]]]:
        model_path = Path(config.model_path)

        if not model_path.exists() or not model_path.is_dir():
            raise ValueError(f'Path "{model_path}" is not a valid path to model')

        if not (model_path / 'model.pkl').exists() or not (model_path / 'vectorizer.pkl').exists():
            raise ValueError(f'Model from "{model_path}" is corrupted')

        if config.top_n <= 0:
            raise ValueError(f'Top n value "{config.top_n}" must be positive int')

        with open(model_path / 'model.pkl', 'rb') as fin:
            model = pickle.load(fin)

        with open(model_path / 'vectorizer.pkl', 'rb') as fin:
            vectorizer = pickle.load(fin)

        scores_list = model.predict_proba(vectorizer.transform(texts))

        predicted = []
        for scores in scores_list:
            sorted_scores = np.sort(scores)[::-1]
            sorted_labels = [model.classes_[i] for i in np.argsort(scores)][::-1]
            predicted.append(list(zip(sorted_labels, sorted_scores))[: config.top_n])

        return predicted

    