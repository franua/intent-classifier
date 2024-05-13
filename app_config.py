import yaml
from yaml.loader import FullLoader


class AppConfig:
    def __init__(self) -> None:
        self._config = None
        with open("./config/app_config.yaml", "r") as f:
            self._config = yaml.load(f, Loader=FullLoader)

    def get_config(self):
        return self._config

    def get_inference_classes(self) -> list:
        return self._config["inference_classes"]
