import fastapi
import logging
import pandas as pd

from challenge.model import DelayModel
from challenge.model_wrapper import ModelWrapper
from challenge.utils import adjust_dummy_columns, get_dummy_representation


model_wrapper = ModelWrapper()
app = fastapi.FastAPI()


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200, response_model=dict)
async def post_predict(request: fastapi.Request, response: fastapi.Response) -> dict:
    prediction_model = ModelWrapper.get_model()
    result = []
    try:
        request_json = await request.json()
    except BaseException as be:
        logging.error("Found error in json input: %s, skipping", str(be))
        response.status_code = 400
        request_json = None
    if request_json:
        dataframe_input = pd.DataFrame(request_json['flights'])
        is_input_ok = DelayModel.validate_input(dataframe_input)
        if is_input_ok:
            features = get_dummy_representation(dataframe_input)
            processed_input = adjust_dummy_columns(features, DelayModel.Top_10_Features)
            predicted_target = prediction_model.predict(processed_input)
            result = [int(x) for x in predicted_target]
        else:
            response.status_code = 400
    return {'predict': result}
