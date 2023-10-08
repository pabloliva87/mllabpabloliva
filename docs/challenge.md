1. Issue: trying to test (using "make api-test") yields
E           AttributeError: module 'anyio' has no attribute 'start_blocking_portal'

Apparently, the version of anyio is important. Downgraded anyio to 3.7.1:
$ pip install anyio==3.7.1
Modified requirements.txt to comply with that requirement. If this wasn't meant to be part of the challenge, please add this version restriction to the requirements.


2. We have four parts:
 * In order to operationalize the model, transcribe the `.ipynb` file into the `model.py` file.

 * Deploy the model in an `API` with `FastAPI` using the `api.py` file.

 * Deploy the `API` in your favorite cloud provider (we recomend to use GCP).

 * We are looking for a proper `CI/CD` implementation for this development.

We'll create a feature branch for each, and update develop when done.

3. Operationalize the model
We'll pick XG Boost with top10 features and class balancing, because its f1-score for delays is marginally better (0.37 vs 0.36 of Regression). In reality, going with either one could be defensible at this stage from a precission point of view.

Tests are failing due to the path to data being invalid. The long term solution is to set an env variable and use that for the tests to locate their data. For now, we'll use a hardcoded path as default.

The test test_model_predict appears to be ill-conceived; it asks for a prediction without having called fit first. It was reworked to do that crucial step first, but keeping all the original assertions.

get_min_diff shows that sometimes, flights leave before their departure time. Not a delay, but certainly interesting. I've set up a warning to be logged if we see a plane leaving an hour or more prior to its departure time. That case would deserve some further study.

The type hint for preprocess needs to use square brackets to work, so it now looks like Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]

4. Deploy the model

Run API by doing
uvicorn challenge.api:app --reload

We'll use a local install of Postman to test the first requests.

It turns out that we need the code that converts data into dummies twice; once for preprocessing and in the api, to transform the received input. We'll modularize it to utils.

The above turned out to be a bit more complicated: for training, we need to get the dummies and reduce them to the subset the model expects; for processing, we need to get the dummies, add the columns the module expects if any are missing, and remove the ones the modules does not expect if any are present. Utils and its tests have grown a bit because of these requirements.


