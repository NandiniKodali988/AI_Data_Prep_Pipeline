FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY src/ ./src/
COPY streamlit_app.py ./
COPY main.py ./

RUN mkdir -p /tmp/uploads /app/output && chmod 777 /tmp/uploads /app/output

EXPOSE 7860

HEALTHCHECK --interval=10s --timeout=10s --start-period=30s --retries=5 CMD curl --fail http://localhost:7860/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
