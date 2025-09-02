FROM python:3.13-slim-bookworm

RUN apt-get update && apt-get install gcc build-essential -y && python -m pip install --upgrade pip

COPY . /app
WORKDIR /app

RUN pip install .

EXPOSE 8080

CMD ["uvicorn", "hashcalc.api:app", "--host", "0.0.0.0", "--port", "8080"]
