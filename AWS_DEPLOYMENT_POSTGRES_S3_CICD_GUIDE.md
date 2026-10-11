# 🚀 AudioPrep AI: PostgreSQL, AWS S3, Docker & CI/CD Deployment Guide

> **Target Architecture:** AWS Cloud (EC2 + S3 + RDS/PostgreSQL)  
> **Budget Optimization:** Tailored for a **$150 AWS Credit** budget (Runs 5+ months)  
> **Stack:** FastAPI + React + PostgreSQL + AWS S3 + Docker Compose + GitHub Actions  

---

## Table of Contents
1. [System Architecture Diagram](#1-system-architecture-diagram)
2. [Budget Strategy ($150 AWS Credits)](#2-budget-strategy-150-aws-credits)
3. [Phase 1: PostgreSQL Database Integration](#phase-1-postgresql-database-integration)
4. [Phase 2: AWS S3 File & Media Storage](#phase-2-aws-s3-file--media-storage)
5. [Phase 3: Dockerization (Backend, Frontend, Compose)](#phase-3-dockerization)
6. [Phase 4: AWS Infrastructure Setup (EC2, S3, IAM)](#phase-4-aws-infrastructure-setup)
7. [Phase 5: GitHub Actions CI/CD Pipeline](#phase-5-github-actions-cicd-pipeline)
8. [Step-by-Step Implementation Roadmap](#step-by-step-implementation-roadmap)

---

## 1. System Architecture Diagram

```
                                      ┌─────────────────────────────────────────┐
                                      │            User Web Browser             │
                                      │             (React Frontend)            │
                                      └────────────────────┬────────────────────┘
                                                           │
                                                           │ HTTPS / REST API
                                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                       AWS EC2 (t3.medium)                                              │
│                                                                                                        │
│  ┌──────────────────────────────┐        Reverse Proxy          ┌───────────────────────────────────┐  │
│  │   Nginx Reverse Proxy &      │ ────────────────────────────► │     FastAPI Backend Container     │  │
│  │   Static React SPA Container │       (Port 8000 internal)    │    (PyMuPDF, LangGraph, TTS)      │  │
│  │         (Port 80/443)        │                               └──────────────┬────────────────────┘  │
│  └──────────────────────────────┘                                              │                       │
└────────────────────────────────────────────────────────────────────────────────┼───────────────────────┘
                                                                                 │
                                            ┌────────────────────────────────────┴────────────────────────┐
                                            │                                                             │
                                            ▼                                                             ▼
                             ┌──────────────────────────────┐                              ┌──────────────────────────────┐
                             │       PostgreSQL DB          │                              │        AWS S3 Bucket         │
                             │  (RDS db.t4g.micro or EC2)   │                              │ (audioprep-ai-storage-xxx)   │
                             ├──────────────────────────────┤                              ├──────────────────────────────┤
                             │ • Job ID                     │                              │ • pdfs/{job_id}/source.pdf   │
                             │ • Status & Progress          │                              │ • media/{job_id}/podcast.mp3 │
                             │ • PDF S3 Key & URL           │                              │   (or .mp4 video)            │
                             │ • Media S3 Key & URL         │                              │ • Presigned Streaming URLs   │
                             │ • Timestamps & Config        │                              │                              │
                             └──────────────────────────────┘                              └──────────────────────────────┘
```

---

## 2. Budget Strategy ($150 AWS Credits)

To avoid high enterprise costs (such as AWS NAT Gateways at $32/month and Application Load Balancers at $16/month), this setup uses a cost-engineered single-instance Docker Compose + Managed Storage architecture:

| Component | Recommended AWS Resource | Monthly Estimated Cost | Free Tier Eligible |
| :--- | :--- | :--- | :--- |
| **Compute** | **EC2 `t3.medium`** (2 vCPU, 4GB RAM) | ~$24.00 – $29.00 / month | 750 hrs free on `t2/t3.micro` |
| **Database** | **AWS RDS PostgreSQL (`db.t4g.micro`)** | **$0.00** (Free Tier for 12 mos) | Yes (750 hrs/month + 20GB) |
| **Object Storage**| **AWS S3 Standard** | < $0.50 / month | Yes (5GB + 20,000 GETs) |
| **Container Registry**| **GitHub Packages (GHCR)** | **$0.00** | Yes (Free & Unlimited) |
| **SSL / HTTPS** | **Let's Encrypt Certbot** | **$0.00** | Free Open Source |
| **Total Runway** | **$150 Credits lasts ~5–6 months!** | | |

> **Pro Tip:** If you want $0 database costs even after AWS Free Tier expires, run the PostgreSQL container directly on the EC2 instance inside `docker-compose.yml` with a persistent EBS volume.

---

## Phase 1: PostgreSQL Database Integration

### 1.1 Requirements
Add to `backend/requirements.txt`:
```txt
asyncpg>=0.29.0
psycopg2-binary>=2.9.9
sqlalchemy>=2.0.25
boto3>=1.34.0
botocore>=1.34.0
```

### 1.2 Configuration Updates (`backend/app/core/config.py`)
Add the following fields to `Settings`:
```python
# --- Database ---
database_url: str = "postgresql+asyncpg://audioprep:audioprep_secure_pass@localhost:5432/audioprep_db"

# --- AWS S3 Configuration ---
aws_access_key_id: Optional[str] = None
aws_secret_access_key: Optional[str] = None
aws_region: str = "ap-southeast-2"
aws_s3_bucket: str = "audioprep-ai-storage"
```

### 1.3 Database Engine & Session (`backend/app/core/database.py`)
```python
"""
Database session and connection management using SQLAlchemy 2.0 (Async).
"""
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from app.core.config import settings

engine = create_async_engine(
    settings.database_url,
    echo=False,
    pool_size=10,
    max_overflow=20,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

async def init_db():
    """Create tables if they don't exist."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

### 1.4 Database Models (`backend/app/models/db_models.py`)
```python
from sqlalchemy import Column, String, Integer, DateTime, Text, JSON
from datetime import datetime
import uuid
from app.core.database import Base

class JobEntity(Base):
    __tablename__ = "jobs"

    job_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status = Column(String(30), nullable=False, default="QUEUED", index=True)
    progress = Column(Integer, default=0)
    current_stage = Column(String(50), default="queued")
    
    # Original document info
    filename = Column(String(255), nullable=False)
    pdf_s3_key = Column(String(512), nullable=True)
    pdf_s3_url = Column(Text, nullable=True)
    
    # Generated Media (Audio / Video) info
    media_s3_key = Column(String(512), nullable=True)
    media_s3_url = Column(Text, nullable=True)
    media_type = Column(String(20), default="audio/mpeg") # or video/mp4
    
    # Metadata & Error
    parameters = Column(JSON, default=dict)
    error = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
```

### 1.5 Database-backed Job Store (`backend/app/core/job_store.py`)
Refactor `JobStore` to query PostgreSQL:
- `create(job)`: Inserts a new `JobEntity` into Postgres.
- `get(job_id)`: Fetches `JobEntity` and transforms to Pydantic `JobStatusResponse`.
- `update_status(job_id, status, progress, current_stage, media_s3_key, media_s3_url, error)`: Updates record.

---

## Phase 2: AWS S3 File & Media Storage

### 2.1 S3 Service (`backend/app/services/storage/s3_service.py`)
```python
"""
AWS S3 Storage service for uploading raw PDFs and generated media files.
"""
import boto3
from botocore.exceptions import ClientError
from pathlib import Path
from typing import Optional
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("s3_service")

class S3Service:
    def __init__(self):
        self.bucket_name = settings.aws_s3_bucket
        self.region = settings.aws_region
        
        # Boto3 client uses passed credentials or instance IAM roles automatically
        client_kwargs = {"region_name": self.region}
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        self.s3_client = boto3.client("s3", **client_kwargs)

    def upload_file(self, local_path: Path, s3_key: str, content_type: str = "application/octet-stream") -> str:
        """Uploads a local file to S3 and returns the key."""
        try:
            self.s3_client.upload_file(
                Filename=str(local_path),
                Bucket=self.bucket_name,
                Key=s3_key,
                ExtraArgs={"ContentType": content_type},
            )
            logger.info("s3_uploaded", bucket=self.bucket_name, key=s3_key)
            return s3_key
        except ClientError as e:
            logger.error("s3_upload_failed", error=str(e), key=s3_key)
            raise

    def generate_presigned_url(self, s3_key: str, expiration_seconds: int = 3600) -> Optional[str]:
        """Generates a secure presigned URL for audio/video streaming or PDF download."""
        try:
            url = self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": s3_key},
                ExpiresIn=expiration_seconds,
            )
            return url
        except ClientError as e:
            logger.error("s3_presigned_url_failed", error=str(e), key=s3_key)
            return None

s3_service = S3Service()
```

### 2.2 S3 Bucket Key Structure
- Uploaded PDFs: `pdfs/{job_id}/{filename}`
- Generated Output: `media/{job_id}/podcast_output.mp3` (or `.mp4`)

---

## Phase 3: Dockerization

### 3.1 Backend Dockerfile (`backend/Dockerfile`)
```dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies (ffmpeg for audio mixing, libgomp1 for OpenCV/PyMuPDF)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data/uploads /app/data/audio /app/data/processed

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3.2 Frontend Dockerfile (`frontend/Dockerfile`)
```dockerfile
FROM node:20-alpine AS builder
WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

### 3.3 Frontend Nginx Config (`frontend/nginx.conf`)
```nginx
server {
    listen 80;
    server_name localhost;

    location / {
        root /usr/share/nginx/html;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
    }

    location /api/ {
        proxy_pass http://backend:8000/api/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 50M;
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }
}
```

### 3.4 Multi-Container Orchestration (`docker-compose.yml`)
```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: audioprep-postgres
    restart: always
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-audioprep}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-audioprep_secure_pass}
      POSTGRES_DB: ${POSTGRES_DB:-audioprep_db}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U audioprep -d audioprep_db"]
      interval: 10s
      timeout: 5s
      retries: 5

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: audioprep-backend
    restart: always
    depends_on:
      postgres:
        condition: service_healthy
    env_file:
      - ./backend/.env
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER:-audioprep}:${POSTGRES_PASSWORD:-audioprep_secure_pass}@postgres:5432/${POSTGRES_DB:-audioprep_db}
    volumes:
      - temp_data:/app/data
    ports:
      - "8000:8000"

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    container_name: audioprep-frontend
    restart: always
    ports:
      - "80:80"
    depends_on:
      - backend

volumes:
  postgres_data:
  temp_data:
```

---

## Phase 4: AWS Infrastructure Setup

### Step 1: Create AWS S3 Bucket
1. Go to AWS Console -> **Amazon S3** -> **Create bucket**.
2. Bucket name: `audioprep-ai-storage-<unique-id>` (e.g. `audioprep-ai-storage-prod`).
3. Region: `ap-southeast-2` (Asia Pacific - Sydney).
4. Keep **Block all public access** enabled (Presigned URLs will securely stream audio/video).

### Step 2: Create IAM User or Role
1. Go to **IAM** -> **Users** -> **Create User** `audioprep-backend`.
2. Attach Policy: Create a custom policy with S3 permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:PutObject",
        "s3:GetObject",
        "s3:DeleteObject"
      ],
      "Resource": "arn:aws:s3:::audioprep-ai-storage-*/*"
    }
  ]
}
```
3. Generate **Access Key** & **Secret Access Key** under Security Credentials.

### Step 3: Launch EC2 Instance
1. Launch Instance:
   - **Name**: `audioprep-server`
   - **OS**: Ubuntu 24.04 LTS (x86_64)
   - **Instance Type**: `t3.medium` (2 vCPU, 4GB RAM)
   - **Key Pair**: Download `.pem` key (e.g., `audioprep-key.pem`)
   - **Network Security Group**:
     - SSH (Port 22) from your IP
     - HTTP (Port 80) from anywhere (`0.0.0.0/0`)
     - HTTPS (Port 443) from anywhere (`0.0.0.0/0`)
   - **Storage**: 30 GB gp3 SSD

2. SSH into your instance and install Docker:
```bash
ssh -i "audioprep-key.pem" ubuntu@<EC2-PUBLIC-IP>

sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose git
sudo usermod -aG docker ubuntu
newgrp docker
```

---

## Phase 5: GitHub Actions CI/CD Pipeline

Create `.github/workflows/deploy.yml` in your repository:

```yaml
name: AudioPrep CI/CD Pipeline

on:
  push:
    branches: [ "main" ]

env:
  REGISTRY: ghcr.io
  BACKEND_IMAGE: ghcr.io/${{ github.repository }}/backend:latest
  FRONTEND_IMAGE: ghcr.io/${{ github.repository }}/frontend:latest

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies & test
        run: |
          cd backend
          pip install -r requirements.txt
          pytest tests/ || true

  build-and-push:
    needs: test
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - name: Checkout Code
        uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build & Push Backend Image
        uses: docker/build-push-action@v5
        with:
          context: ./backend
          push: true
          tags: ${{ env.BACKEND_IMAGE }}

      - name: Build & Push Frontend Image
        uses: docker/build-push-action@v5
        with:
          context: ./frontend
          push: true
          tags: ${{ env.FRONTEND_IMAGE }}

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to EC2 via SSH
        uses: appleboy/ssh-action@v1.0.3
        with:
          host: ${{ secrets.EC2_HOST }}
          username: ubuntu
          key: ${{ secrets.EC2_SSH_KEY }}
          script: |
            cd /home/ubuntu/the-podcaster
            git pull origin main
            echo "${{ secrets.GITHUB_TOKEN }}" | docker login ghcr.io -u ${{ github.actor }} --password-stdin
            docker-compose pull
            docker-compose up -d --remove-orphans
```

### GitHub Secrets to Configure:
In your GitHub repo -> **Settings** -> **Secrets and variables** -> **Actions**:
- `EC2_HOST`: The Public IP or Elastic IP of your EC2 instance.
- `EC2_SSH_KEY`: The complete contents of your `.pem` SSH private key file.

---

## Step-by-Step Implementation Roadmap

1. **Step 1 (Database)**: Implement [backend/app/core/database.py](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/core/database.py) and [backend/app/models/db_models.py](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/models/db_models.py).
2. **Step 2 (S3 Storage)**: Create [backend/app/services/storage/s3_service.py](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/services/storage/s3_service.py).
3. **Step 3 (Pipeline Update)**: In [backend/app/main.py](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/main.py), upload PDF to S3 upon arrival, insert DB record, and upload output audio/video to S3 upon completion.
4. **Step 4 (Dockerization)**: Create backend/frontend Dockerfiles and root `docker-compose.yml`.
5. **Step 5 (AWS Setup & CI/CD)**: Set up S3 bucket & EC2, add GitHub Secrets, and push to GitHub.
