FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUTODRIVE_RUNTIME_ROOT=/var/lib/autodrive-dataops \
    AUTODRIVE_STATE_DIR=/var/lib/autodrive-dataops/state \
    AUTODRIVE_DB_PATH=/var/lib/autodrive-dataops/state/autodrive_state.sqlite \
    AUTODRIVE_CHECKPOINT_PATH=/var/lib/autodrive-dataops/state/checkpoints.sqlite \
    AUTODRIVE_CHECKPOINT_BACKEND=sqlite \
    API_HOST=0.0.0.0 \
    API_PORT=8080

WORKDIR /opt/autodrive
COPY pyproject.toml README.md ./
COPY deploy_ci_cloud_agentv3 ./deploy_ci_cloud_agentv3

RUN python -m pip install --no-cache-dir . \
    && mkdir -p /var/lib/autodrive-dataops/state /var/lib/autodrive-dataops/config /var/lib/autodrive-dataops/data \
    && useradd --system --uid 10001 --create-home autodrive \
    && chown -R autodrive:autodrive /var/lib/autodrive-dataops

USER autodrive
VOLUME ["/var/lib/autodrive-dataops"]
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).read()" || exit 1
ENTRYPOINT ["autodrive-agent"]
CMD ["serve"]
