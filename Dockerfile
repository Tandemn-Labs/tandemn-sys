# Runs Orca and the GPU metrics collector (entry points: tandemn-orca,
# tandemn-gpu-metrics-collector, tandemn-submit-job).
#
# tandemn-orca depends on the sibling tandemn-store repo (a path dependency),
# so build from the workspace directory that contains BOTH repos:
#
#   docker build -f tandemn-system/Dockerfile -t tandemn-system:latest .

FROM python:3.12-slim

WORKDIR /app

COPY tandemn-store/ tandemn-store/
COPY tandemn-system/ tandemn-system/

# Install the store first so the "tandemn-store" dependency of tandemn-orca
# resolves locally instead of from an index.
RUN pip install --no-cache-dir ./tandemn-store && \
    pip install --no-cache-dir ./tandemn-system

# Overridden per Deployment (orca vs collector).
CMD ["tandemn-orca"]
