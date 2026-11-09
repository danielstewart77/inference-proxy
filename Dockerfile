FROM python:3.11-slim

WORKDIR /app

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application ----
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini ./

ENV PYTHONUNBUFFERED=1
EXPOSE 8888

CMD ["python", "-m", "app.entrypoint"]
