1. Issue: trying to test (using "make api-test") yields
E           AttributeError: module 'anyio' has no attribute 'start_blocking_portal'

Apparently, the version of anyio is important. Downgraded anyio to 3.7.1:
$ pip install anyio==3.7.1
Modified requirements.txt to comply with that requirement. If this wasn't meant to be part of the challenge, please add this version restriction to the requirements.


