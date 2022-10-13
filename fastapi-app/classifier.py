import pickle
import shutil
import numpy as np

from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from structures import FitConfig, PredictConfig, Texts, Labels, Scores, Prediction

class TextClassifier:
    @staticmethod
    def fit(texts: Texts, labels: Labels, config: FitConfig) -> None:
        if config.feature_type == 'tf-idf':
            vectorizer = TfidfVectorizer()

        elif config.feature_type == 'bow':
            vectorizer = CountVectorizer()

        else:
            raise ValueError(f'Unknown config.feature_type: "{config.feature_type}"')

        data = vectorizer.fit_transform(texts.values)
        model = LogisticRegression()
        model.fit(data, labels.values)

        model_path = Path(config.model_path)

        if model_path.exists():
            shutil.rmtree(model_path)
        model_path.mkdir()

        with open(model_path / 'model.pkl', 'wb') as fout:
            pickle.dump(model, fout)

        with open(model_path / 'vectorizer.pkl', 'wb') as fout:
            pickle.dump(vectorizer, fout)

    @staticmethod
    def predict(texts: Texts, config: PredictConfig) -> Prediction:
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

        scores_list = model.predict_proba(vectorizer.transform(texts.values))

        labels_list_ = []
        scores_list_ = []

        for scores in scores_list:
            sorted_scores = list(np.sort(scores))[::-1]
            sorted_labels = [model.classes_[i] for i in np.argsort(scores)][::-1]
            labels_list_.append(sorted_labels[: config.top_n])
            scores_list_.append(sorted_scores[: config.top_n])

        return Prediction(
            labels_list=[Labels(values=labels) for labels in labels_list_],
            scores_list=[Scores(values=scores) for scores in scores_list_],
        )