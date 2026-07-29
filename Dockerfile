FROM python:3.11-slim

WORKDIR /app

# ---- Python deps ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Application ----
COPY app/ ./app/
COPY migrations/ ./migrations/
COPY alembic.ini ./

# SQLite lives here. Mount a volume over it to keep the database across
# rebuilds; the path is overridable with DB_PATH.
VOLUME ["/app/data"]
ENV DB_PATH=/app/data/proxy.db

ENV PYTHONUNBUFFERED=1
EXPOSE 8888

CMD ["python", "-m", "app.entrypoint"]
