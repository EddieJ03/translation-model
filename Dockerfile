FROM python:3.10-slim
WORKDIR /my-modelaugbbqezdqix
STOPSIGNAL SIGINT

ENV LISTEN_PORT 8000

# System dependencies
RUN apt update && apt install -y libgomp1
RUN pip3 install poetry

# Project dependencies
COPY . .

RUN poetry config virtualenvs.create false
RUN poetry config installer.parallel false
RUN poetry install --no-interaction --no-ansi --only main

ENTRYPOINT uvicorn my-modelaugbbqezdqix.serving.serve:app --host 0.0.0.0 --port $LISTEN_PORT --workers 2