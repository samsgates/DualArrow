FROM python:3.14-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DUALARROW_CONFIG=/app/dualarrow.yaml

RUN addgroup --system dualarrow && adduser --system --ingroup dualarrow dualarrow
WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY dualarrow ./dualarrow
RUN python -m pip install --upgrade pip && \
    python -m pip install ".[all-adapters]"

COPY examples/quickstart/dualarrow.yaml /app/dualarrow.yaml
COPY examples/quickstart/knowledge /app/knowledge
RUN mkdir -p /app/data && chown -R dualarrow:dualarrow /app

USER dualarrow
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3)"

CMD ["dualarrow", "serve", "--config", "/app/dualarrow.yaml"]
