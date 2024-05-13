from logging_config import configure_logging


def pytest_sessionstart(session):
    configure_logging()
