"""ASGI entrypoint for the `models` service (design §5.1, plan 15 task 6).

The Dockerfiles' CMD runs this as a uvicorn factory:
    uvicorn modelsvc.asgi:app_factory --factory --host 0.0.0.0 --port 9000
"""

from config import get_settings
from modelsvc.app import create_models_app
from modelsvc.backends import build_backend


def app_factory():
    return create_models_app(build_backend(get_settings()))
