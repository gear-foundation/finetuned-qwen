FROM python:3.9-slim

WORKDIR /app

RUN apt update && apt install -y wget unzip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install gdown && \
    gdown --id 1kigjQ0rBYyeGqzomWhaFBhSpgjB8fBe5 -O model.zip && \
    unzip model.zip -d qwen-finetuned && \
    rm model.zip && \
    gdown --id 1cOOiQVKtyntYvWMC0JZBiXp9At0GbJty -O model.zip && \
    unzip model.zip -d bert-model && \
    rm model.zip

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
