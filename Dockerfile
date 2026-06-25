# syntax=docker/dockerfile:1.7

############################
# Builder
############################
FROM python:3.13-slim AS builder

WORKDIR /workspace/train_boundless_das

# Build dependencies only needed during install
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:${PATH}"

# Copy dependency manifests first for maximum cache reuse
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
    --frozen \
    --no-dev

# Copy source code after dependency layer
COPY conf/ conf/
COPY tutorial_price_tagging_utils.py .
COPY generate_data.py .
COPY boudless_das.py .

############################
# Runtime
############################
FROM python:3.13-slim AS runtime

WORKDIR /workspace/train_boundless_das

# Runtime packages only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy uv binary
COPY --from=builder /root/.local/bin/uv /usr/local/bin/uv

# Copy virtual environment
COPY --from=builder /workspace/train_boundless_das/.venv \
    /workspace/train_boundless_das/.venv

# Copy application code
COPY --from=builder /workspace/train_boundless_das/conf ./conf
COPY --from=builder /workspace/train_boundless_das/tutorial_price_tagging_utils.py .
COPY --from=builder /workspace/train_boundless_das/generate_data.py .
COPY --from=builder /workspace/train_boundless_das/boudless_das.py .

ENV PATH="/workspace/train_boundless_das/.venv/bin:${PATH}"

CMD ["python", "generate_data.py"]