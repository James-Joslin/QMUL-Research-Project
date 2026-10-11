# Development image only; source is mounted by Compose for editable iteration.
FROM python:3.12.15-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.13.0 /uv /uvx /usr/local/bin/
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG ACCELERATOR=cpu
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential swig git graphviz libgl1 libglib2.0-0 libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid ${GROUP_ID} researcher \
    && useradd --create-home --uid ${USER_ID} --gid ${GROUP_ID} researcher \
    && mkdir -p /workspace /opt/venv /opt/uv-cache \
    && chown -R researcher:researcher /workspace /opt/venv /opt/uv-cache
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_CACHE_DIR=/opt/uv-cache \
    UV_PYTHON_DOWNLOADS=0 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" MPLBACKEND=Agg SDL_AUDIODRIVER=dummy
WORKDIR /workspace
COPY --chown=researcher:researcher pyproject.toml uv.lock README.md LICENSE ./
COPY --chown=researcher:researcher src ./src
USER researcher
RUN uv sync --locked --extra ${ACCELERATOR} --extra analysis --extra tuning --extra diagnostics
CMD ["bash"]
