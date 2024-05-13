# -*- coding: utf-8 -*-

import os
import logging
from enum import Enum
import torch
from transformers import Pipeline, ZeroShotClassificationPipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Optional
from dataclasses import dataclass
from response import Prediction, PredictionResponse


class PipelineType(Enum):
    ZERO_SHOT = "zero-shot"


@dataclass
class ModelArtifact:
    model: Optional[AutoModelForSequenceClassification] = None
    tokenizer: Optional[AutoTokenizer] = None

    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None


class IntentClassifier:
    def __init__(self):
        self.__pipeline = None

    def is_ready(self):
        return isinstance(self.__pipeline, Pipeline)

    def infer_top_3_classes_zs(
        self, text: str, canditate_labels: list
    ) -> list[Prediction]:
        if not isinstance(self.__pipeline, ZeroShotClassificationPipeline):
            error_message = f"Attempt to infer using a ZERO-SHOT pipeline when '{type(self.__pipeline)}' is loaded."
            raise ValueError(error_message)

        inferred_classes = self.__pipeline(text, canditate_labels)

        # no need to sort, `labels` and `scores` are sorted in descending order of scores by default
        top_3_classes = []
        labels = inferred_classes["labels"]
        scores = inferred_classes["scores"]

        # select the top 3 classes
        for label, score in zip(labels[:3], scores[:3]):
            prediction = Prediction(label=label, score=score)
            top_3_classes.append(prediction)

        logging.info(f"Inferred {top_3_classes} for the `text`: '{text}'.")

        return top_3_classes

    def load(
        self,
        model_name: str = "typeform/distilbert-base-uncased-mnli",
        model_dir: str = "./models",
        pipeline_type: str = PipelineType.ZERO_SHOT,
    ) -> Optional[Pipeline]:
        """
        The method loads up the model and tokenizer files in memory and builds a transformers.Pipeline of a specified type which is stored in `_pipeline` and returned.
        If `_pipeline` is already set and its model's name equals requested `model_name` - the `_pipeline` is returned.
        If you want to `load()` another model - explicitly call `release()` first.
        ! Note: after the `release()` is called and until the next `load()` is succeded, `is_ready()` will be returning `False`.
        """

        if self._is_model_loaded(model_name):
            logging.debug(
                f"Requested model '{model_name}' is already loaded. Returning the pipeline."
            )
            return self.__pipeline

        os.makedirs(model_dir, exist_ok=True)

        model_artifact = self._load_artifact(model_dir=model_dir, model_name=model_name)
        if model_artifact.loaded():
            logging.info(f"The model '{model_name}' has been successfully loaded!")
            self.__pipeline = self._pipeline_factory(model_artifact, pipeline_type)

        return self.__pipeline

    def unload(self):
        """
        Unloads the `_pipeline` from memory.
        """
        logging.warning(f"Unloading the current model pipeline.")
        self.__pipeline = None

    def _is_model_loaded(self, model_name: str) -> bool:
        return (
            isinstance(self.__pipeline, Pipeline)
            and self.__pipeline.model.name_or_path == model_name
        )

    def _load_artifact(self, model_dir: str, model_name: str) -> ModelArtifact:
        """
        The method will attempt to load pre-cached 🤗 model and its tokenizer from the specified caching directory `model_dir` by the `model_name`.
        If the specified model's not found in the `model_dir` - it will be downloaded from HF and stored in that directory.
        Returns a ModelArtifact object.
        """
        try:
            model_artifact = ModelArtifact(
                model=AutoModelForSequenceClassification.from_pretrained(
                    model_name, cache_dir=model_dir
                ),
                tokenizer=AutoTokenizer.from_pretrained(
                    model_name, cache_dir=model_dir
                ),
            )
            return model_artifact
        except Exception as e:
            logging.error(
                f"Error: Unable to load model '{model_name}' from 🤗 or load cached from '{model_dir}'.\nException: {e}."
            )
            return ModelArtifact()

    def _pipeline_factory(
        self,
        model_artifact: ModelArtifact,
        pipeline_type: str = PipelineType.ZERO_SHOT,
    ) -> Pipeline:
        match pipeline_type:
            case PipelineType.ZERO_SHOT:
                return ZeroShotClassificationPipeline(
                    model=model_artifact.model,
                    tokenizer=model_artifact.tokenizer,
                    device=self._determine_device(),
                )
            case _:
                raise ValueError(f"Unsupported pipeline type: {pipeline_type}")

    def _build_model_artifact_path(self, model_dir: str, model_name: str):
        return os.path.join(model_dir, model_name.replace("/", "--") + ".pt")

    def _determine_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


if __name__ == "__main__":
    pass
