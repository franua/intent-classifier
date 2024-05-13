# Use the official Python image as a base image
FROM python:3.11-slim

# Set environment variables for Python buffering
ENV PYTHONUNBUFFERED 1

WORKDIR /app

# Install build tools and dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY Pipfile Pipfile.lock /app/

RUN pip install pipenv \
    && pipenv install --deploy --system

COPY . /app/

EXPOSE 8080

CMD ["python", "server.py"]
