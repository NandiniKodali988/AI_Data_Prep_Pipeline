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
COPY chroma_db/ ./chroma_db/
COPY streamlit_app.py ./
COPY main.py ./

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
