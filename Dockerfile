FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /opt/adm-monai-unet

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md MANIFEST.in ./
COPY src ./src
COPY examples ./examples

RUN python -m pip install --upgrade pip && \
    python -m pip install ".[monai]"

CMD ["python", "examples/monai_style_usage.py"]
