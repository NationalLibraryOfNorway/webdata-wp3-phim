FROM python:3.13-slim-bookworm

COPY . /app
WORKDIR /app

RUN pip install --upgrade pip && pip install .

EXPOSE 8080

CMD ["uvicorn", "hashcalc.api:app", "--host", "0.0.0.0", "--port", "8080"]
