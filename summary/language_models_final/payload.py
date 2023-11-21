import enum
import traceback

from pydantic import ValidationError


class ResultTypes(enum.Enum):
    DONE = enum.auto()
    USER_ERROR = enum.auto()
    APPLICATION_ERROR = enum.auto()
    DISCONNECTED = enum.auto()


def validation_exception(exc):
    errors = exc.errors()
    for entry in errors:
        entry["loc"] = entry["loc"][1:]
    return {
        "error": "VALIDATION",
        "errors": errors,
    }


def general_exception(exc):
    return {
        "error": "APPLICATION",
        "message": str(exc),
    }


def exception_to_payload(exc):
    if isinstance(exc, ValidationError):
        result = validation_exception(exc), 422
    else:
        print(traceback.format_exc())
        result = general_exception(exc), 500
    result[0]["success"] = False
    return result


def processed_to_payload(result_type, result=None):
    if result_type == ResultTypes.DONE:
        return {"success": True, "data": result}, 200
    elif result_type == ResultTypes.USER_ERROR:
        return {"success": False, "error": "USER", "message": result}, 400
    elif result_type == ResultTypes.APPLICATION_ERROR:
        return {"success": False, "error": "APPLICATION", "message": result}, 500
    elif result_type == ResultTypes.DISCONNECTED:
        return {
            "success": False,
            "error": "DISCONNECTED",
            "message": "connection lost",
        }, 204
    return {
        "success": False,
        "error": "APPLICATION",
        "message": "request was not done and had no errors",
    }, 500
