import importlib
from os import environ

try:
    LANGUAGE_MODEL = environ["LANGUAGE_MODEL"]
except KeyError:
    print("Environment variable 'LANGUAGE_MODEL' is not defined.")
    print(
        "Set it to the file name of the model from the './models' folder without the .py extension."
    )
    exit(1)

Model = importlib.import_module(f"models.{LANGUAGE_MODEL}").Model

model = Model()

DEFAULT_SETTINGS = {
    "threads": 4,
    "batch_size": 1,
    "cache_size": 0,
}

try:
    DEFAULT_SETTINGS = {
        int(model.PREFERRED_SETTINGS.get(key, value))
        for key, value in DEFAULT_SETTINGS.items()
    }
except ValueError:
    print(
        f"One of the settings of the PREFERRED_SETTINGS dict from {LANGUAGE_MODEL}.py is not an int"
    )
except AttributeError as e:
    pass

SETTINGS = {}

for key, value in DEFAULT_SETTINGS.items():
    ENVIRONMENT_VARIABLE = key.upper()
    try:
        new_value = environ.get(ENVIRONMENT_VARIABLE, value)
        SETTINGS[key] = int(new_value)
    except ValueError:
        print(f"{ENVIRONMENT_VARIABLE} is provided but the value is not an int")
        print(f"{ENVIRONMENT_VARIABLE}: {new_value}")
        exit(1)

from typing import List, Tuple

from application import FuncFastAPI
from argument_models import create_function_validator

MODEL_TYPES = {
    "generation": [("batch", List[str])],
    "metric": [("batch", List[Tuple[str, str]])],
}

try:
    positional_arguments = MODEL_TYPES[model.TYPE]
except KeyError:
    print(f"The ./models/{LANGUAGE_MODEL}.py file contains errors.")
    print("The TYPE variable on the Model class was not set.")
    print(f"Set it to one of {list(MODEL_TYPES.keys)}")
    exit(1)

_, _, _, full_validator = create_function_validator(
    model.__call__, positional_arguments=positional_arguments
)

SETTINGS["model_name"] = LANGUAGE_MODEL
app = FuncFastAPI(model.__call__, full_validator, **SETTINGS)

# the following line does not work for some reason:
# uvicorn.run(app, host="0.0.0.0", port=5000)
# use the following commandline instruction instead:
# uvicorn app:app --host 0.0.0.0 --port 5000
