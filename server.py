# -*- coding: utf-8 -*-

import argparse
from app_config import AppConfig
from flask import Flask
from flask import request, jsonify
from flasgger import Swagger, swag_from
from intent_classifier import IntentClassifier
import logging
from logging_config import configure_logging
import os
from response import PredictionResponse, ErrorResponse, swagger_specs_dict

app = Flask(__name__)
model = IntentClassifier()
app_config = AppConfig()
swagger = Swagger(app)


@app.route("/ready", methods=["GET"])
def ready():
    """
    Check if the model is ready.
    ---
    responses:
      200:
        description: OK if the model is ready
      503:
        description: Not ready if the model is not ready
    """
    if model.is_ready():
        return "OK", 200
    else:
        # return "Not ready", 423
        # The 4xx group of HTTP-codes is called 'Client Error', which is not the case here
        # imo 503 Service Unavailable semantically suites the situation better
        # also "model's not ready" can be a legit incident / errorneus situation caused by network, IO and whatnot issue
        return "Not ready", 503


@app.route("/intent", methods=["POST"])
@swag_from(swagger_specs_dict)
def intent():
    """
    Predict the intent of the given text.
    ---
    parameters:
      - in: body
        name: text
        description: The text to predict the intent for
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              description: The text to predict the intent for
    responses:
      200:
        description: Successful prediction
        schema:
          $ref: '#/definitions/PredictionResponse'
      400:
        description: Bad request
        schema:
          $ref: '#/definitions/ErrorResponse'
      500:
        description: Internal server error
        schema:
          $ref: '#/definitions/ErrorResponse'
    """

    # check if request has a body
    if not request.json:
        return (
            jsonify(
                ErrorResponse(
                    label="BODY_MISSING", message="Request doesn't have a body."
                )
            ),
            400,
        )

    # check that `text` is present
    data = request.json
    if "text" not in data or len(data["text"]) == 0:
        return (
            jsonify(
                ErrorResponse(
                    label="TEXT_MISSING",
                    message='"text" missing from request body.',
                )
            ),
            400,
        )

    text = data["text"]
    # validate the length of the text
    # when experimenting with ATIS dataset, we saw that max `text` words was 46 and max `text` length was 259
    # with that, let's initially limit expected input to those numbers + 20% roughly
    if len(text.split()) > 60 or len(text) > 310:
        return (
            jsonify(
                ErrorResponse(label="TEXT_TOO_LONG", message='"text" is too long.')
            ),
            400,
        )

    try:
        top_3_classes = model.infer_top_3_classes_zs(
            text=text, canditate_labels=app_config.get_inference_classes()
        )

        return jsonify(PredictionResponse(top_3_classes))
    except Exception as e:
        logging.error(e)
        return jsonify(ErrorResponse(label="INTERNAL_ERROR", message=str(e))), 500


def main():
    configure_logging()
    logging.info(f"App config: {app_config.get_config()}")
    logging.info(f"Inference classes: {app_config.get_inference_classes()}")

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--model_name",
        type=str,
        default="typeform/distilbert-base-uncased-mnli",
        help="The name of a pre-trained model to load (e.g. 'typeform/distilbert-base-uncased-mnli').",
    )
    arg_parser.add_argument(
        "--port", type=int, default=os.getenv("PORT", 8080), help="Server port number."
    )
    args = arg_parser.parse_args()

    # we need to load up the model BEFORE starting the HTTP-service
    model.load(args.model_name)

    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
