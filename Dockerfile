# Shared Dockerfile for all 4 services (Fast / Balanced / Heavy / Gateway).
# The specific service to run is chosen via the CMD override in docker-compose.yml,
# so we only need ONE Dockerfile for all of them -- same dependencies, same code,
# different entrypoint command.
#
# Build context MUST be the project root (D:\Projects\inferpilot), not serving/,
# because we need access to serving/, training/ (models + checkpoints), and
# router/ (the trained router pickle used by the gateway).
#
# Uses a CUDA-enabled base image so torch.cuda.is_available() returns True
# inside the container -- requires --gpus all at runtime (handled via
# docker-compose.yml's deploy.resources.reservations.devices block).

FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python 3.11 + pip + system libs.
# libglib2.0-0 added as a safety net for opencv-python-headless -- some
# minimal images throw an import-time .so error without it even with the
# headless build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    libjpeg-turbo8 \
    zlib1g \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.11 /usr/bin/python

# Install Python deps first (better layer caching -- only reinstalls if
# requirements_docker.txt changes, not on every code edit).
#
# Uses a BuildKit cache mount for pip's download cache. This means that even
# when requirements_docker.txt changes and this layer has to rerun, pip pulls
# already-downloaded wheels (torch, torchvision, etc.) from the persistent
# cache instead of re-downloading them from the network. This is what makes
# future dependency additions fast instead of re-downloading everything.
# (Requires BuildKit, which is on by default in modern Docker Desktop.)
COPY requirements_docker.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install -r requirements_docker.txt

# Copy only what serving actually needs:
# - serving/ (the FastAPI apps, including the gateway)
# - training/models/ (architecture code)
# - training/checkpoints/ (.pt weight files)
# - router/router_best_model.pkl (the trained router, needed by the gateway)
# NOT the raw dataset, training scripts, utility-label CSVs, or data/ folders
# -- not needed at serving time and would bloat the image for no reason.
COPY serving/ ./serving/
COPY training/models/ ./training/models/
COPY training/checkpoints/ ./training/checkpoints/
COPY training/__init__.py ./training/__init__.py
COPY router/router_best_model.pkl ./router/router_best_model.pkl
COPY router/router_singleshot_model.pkl ./router/router_singleshot_model.pkl

# Actual startup command is overridden per-service in docker-compose.yml
EXPOSE 8000 8001 8002 8003