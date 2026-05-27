FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1
ENV SDL_VIDEODRIVER=dummy

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        swig \
        libgl1 \
        libglib2.0-0 \
        libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY CircEnv ./CircEnv

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.5.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124 \
    && pip install --no-cache-dir -e ./CircEnv

WORKDIR /app/CircEnv

CMD ["python", "train_hockey.py"]
