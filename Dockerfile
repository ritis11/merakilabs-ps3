FROM python:3.11-slim

# uv binary
COPY --from=ghcr.io/astral-sh/uv:0.5.4 /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy app
COPY backend ./backend
COPY frontend ./frontend

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000 8501

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
