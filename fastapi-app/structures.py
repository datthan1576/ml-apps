from typing import List, Literal, Union, Tuple, Optional
from pydantic import BaseModel, constr, PositiveInt, validator

class FitConfig(BaseModel):
    model_path: constr(min_length=1)
    feature_type: Union[Literal['tf-idf'], Literal['bow']]

class PredictConfig(BaseModel):
    model_path: constr(min_length=1)
    top_n: PositiveInt

class Texts(BaseModel):
    texts: List[str]

class Labels(BaseModel):
    labels: List[str]

class Scores(BaseModel):
    scores: List[float]

class Prediction(BaseModel):
    labels_list: List[Labels]
    scores_list: List[Scores]

class ReturnValue(BaseModel):
    success: bool
    message: Optional[str]
    traceback: Optional[str]

class PredictReturnValue(ReturnValue):
    prediction: Optional[Prediction]