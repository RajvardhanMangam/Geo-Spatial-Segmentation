FROM python:3.11-slim

# Install GDAL and other system dependencies required by rasterio, shapely, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    g++ \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL env vars so rasterio can find the libs
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

### 🔴 Issue 3 — Backend `requirements.txt` likely missing `redis` async client

The app uses Redis pub/sub. The async client `redis[asyncio]` (or `aioredis`) must be explicitly listed.

**Verify/fix `backend/requirements.txt`** — make sure it includes:
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
rasterio>=1.3.10
shapely>=2.0.4
numpy>=1.26.0
scipy>=1.13.0
pillow>=10.3.0
transformers>=4.40.0
torch>=2.2.0
redis[asyncio]>=5.0.3
aiofiles>=23.2.1
python-dotenv>=1.0.1
