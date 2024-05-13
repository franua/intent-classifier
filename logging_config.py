import logging.config


def configure_logging():
    logging.config.dictConfig(
        {
            "version": 1,
            "formatters": {
                "default_formatter": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                },
            },
            "handlers": {
                "console_handler": {
                    "class": "logging.StreamHandler",
                    "formatter": "default_formatter",
                },
            },
            "loggers": {
                "": {
                    "handlers": ["console_handler"],
                    "level": "DEBUG",
                },
            },
        }
    )
    logging.root.setLevel(logging.DEBUG)
