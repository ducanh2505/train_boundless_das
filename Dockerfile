FROM python:3.13-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
git \
wget \
ffmpeg \
build-essential \
&& rm -rf /var/lib/apt/lists/*
WORKDIR /workspace/train_boundless_das

COPY tutorial_price_tagging_utils.py .
COPY raw_Boundless_DAS.py .
COPY uv.lock .
COPY pyproject.toml uv.lock ./

ENV PATH="/root/.local/bin:$PATH"
RUN uv sync
RUN uv add huggingface_hub

CMD ["uv", "run", "python", "boudless_das.py"]