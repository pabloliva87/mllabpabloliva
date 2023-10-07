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
