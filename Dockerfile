FROM python:3.13-slim

WORKDIR /code

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
  pip install --no-cache-dir --upgrade pip \
  && pip install --prefer-binary -r requirements.txt

COPY ./app ./app

EXPOSE 4000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "4000"]