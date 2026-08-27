FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    AUTODRIVE_RUNTIME_ROOT=/var/lib/autodrive-dataops

WORKDIR /opt/autodrive
COPY pyproject.toml README.md ./
COPY deploy_ci_cloud_agentv3 ./deploy_ci_cloud_agentv3

RUN python -m pip install --no-cache-dir . \
    && mkdir -p /var/lib/autodrive-dataops/config /var/lib/autodrive-dataops/data \
       /var/lib/autodrive-dataops/state /var/lib/autodrive-dataops/logs \
       /var/lib/autodrive-dataops/run /var/lib/autodrive-dataops/secrets \
    && useradd --system --uid 10001 --create-home autodrive \
    && chown -R autodrive:autodrive /var/lib/autodrive-dataops

USER autodrive
VOLUME ["/var/lib/autodrive-dataops"]
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD autodrive-agent ready || exit 1
ENTRYPOINT ["autodrive-agent"]
CMD ["ready"]
