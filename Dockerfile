FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openmpi-bin libopenmpi-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

CMD ["python", "experiments/comparar_experimentos.py", "--reps", "3"]
