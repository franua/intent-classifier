from dataclasses import dataclass


@dataclass
class Prediction:
    label: str
    score: float

    def to_json(self):
        return {"label": self.label, "score": self.score}


@dataclass
class PredictionResponse:
    intents: list[Prediction]

    def to_json(self):
        return {"intents": [intent.to_json() for intent in self.intents]}


@dataclass
class ErrorResponse:
    label: str
    message: str

    def to_json(self):
        return {"label": self.label, "message": self.message}


swagger_specs_dict = {
    "parameters": [
        {
            "name": "text",
            "in": "body",
            "description": "The text to predict the intent for",
            "required": True,
            "schema": {"type": "object", "properties": {"text": {"type": "string"}}},
        }
    ],
    "definitions": {
        "Prediction": {
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "score": {"type": "number", "format": "float"},
            },
        },
        "PredictionResponse": {
            "type": "object",
            "properties": {
                "intents": {
                    "type": "array",
                    "items": {"$ref": "#/definitions/Prediction"},
                }
            },
        },
        "ErrorResponse": {
            "type": "object",
            "properties": {"label": {"type": "string"}, "message": {"type": "string"}},
        },
    },
    "responses": {
        "200": {
            "description": "Successful prediction",
            "schema": {"$ref": "#/definitions/PredictionResponse"},
        },
        "400": {
            "description": "Bad request",
            "schema": {"$ref": "#/definitions/ErrorResponse"},
        },
        "500": {
            "description": "Internal server error",
            "schema": {"$ref": "#/definitions/ErrorResponse"},
        },
    },
}
