<div align="center">

# 🎙️ AudioPrep AI
### Turn Technical Documents & Notes into Interactive Multi-Speaker AI Podcasts

[![Live Demo](https://img.shields.io/badge/Live%20Demo-audioprep--ai.duckdns.org-00C7B7?style=for-the-badge&logo=google-chrome&logoColor=white)](https://audioprep-ai.duckdns.org)
[![AWS](https://img.shields.io/badge/AWS-EC2%20%7C%20S3%20%7C%20ap--southeast--2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white)](https://aws.amazon.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React%2018-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful%20Agents-FF4F00?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose%20%7C%20Multi--Stage-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20Alpine-4169E1?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/12ATHARAV/AudioPrep-AI/actions)
[![SSL](https://img.shields.io/badge/SSL-Let's%20Encrypt%20Certbot-003A70?style=for-the-badge&logo=letsencrypt&logoColor=white)](https://letsencrypt.org/)

<p align="center">
  <b>AudioPrep AI</b> is an enterprise-grade, RAG-grounded system that transforms complex technical PDF documentation into engaging, natural conversational podcasts featuring two AI co-hosts. Built from document ingestion to automated cloud deployment.
</p>

[Explore Features](#-key-features) • [System Architecture](#-end-to-end-architecture) • [AWS & DuckDNS Deployment](#-cloud-deployment--networking-aws--duckdns) • [Quickstart](#-local-development-setup) • [CI/CD Workflow](#-cicd-pipeline-github-actions)

</div>

---

## 📋 Table of Contents
1. [Overview & Philosophy](#-overview--philosophy)
2. [End-to-End Architecture](#-end-to-end-architecture)
3. [Key Features](#-key-features)
4. [Deep-Dive Component Workflows](#-deep-dive-component-workflows)
   - [Document Ingestion (PyMuPDF)](#1-document-ingestion-pymupdf--ocr)
   - [Chunking Strategies Engine](#2-multi-strategy-chunking-engine)
   - [Hybrid Retrieval & Cross-Encoder Reranking](#3-hybrid-dense--sparse-retrieval-qdrant--bm25)
   - [LangGraph Agentic Orchestration](#4-langgraph-agentic-orchestration)
   - [Dual-Engine TTS & Audio Post-Processing](#5-dual-engine-tts--ffmpeg-audio-engineering)
   - [PostgreSQL & AWS S3 Storage Layer](#6-postgresql-state--aws-s3-object-storage)
5. [Cloud Deployment & Networking (AWS + DuckDNS + SSL)](#-cloud-deployment--networking-aws--duckdns)
   - [The DuckDNS & Dynamic DNS Concept](#why-duckdns--dynamic-dns-explained)
   - [Let's Encrypt SSL/TLS Integration](#lets-encrypt-automated-ssltls-certbot)
   - [AWS Infrastructure Details](#aws-infrastructure-cost-engineered)
6. [CI/CD Pipeline (GitHub Actions)](#-cicd-pipeline-github-actions)
7. [Repository Structure](#-project-directory-structure)
8. [Local Development Setup](#-local-development-setup)
9. [API Documentation](#-api-endpoints)
10. [Evaluation Framework (RAGAS)](#-ragas-evaluation-framework)
11. [License & Acknowledgements](#-license)

---

## 💡 Overview & Philosophy

Engineers and students spend hours reading technical papers, release notes, and documentation during busy schedules. **AudioPrep AI** bridges this gap by enabling auditory learning with natural, dialogue-driven discussions between an inquisitive **Host** and an authoritative **Expert**.

Unlike simple "prompt-and-read" wrappers:
- **No Hallucination**: RAG-grounded dialogue generation backed by hybrid dense/sparse search and cross-encoder reranking.
- **Closed-Loop Verification**: A Critic Node validates groundedness, coverage, and conversational pacing before audio synthesis.
- **Production Resilience**: Multi-tier LLM and TTS fallback chains (NVIDIA/DeepSeek ➔ Gemini ➔ Groq; Gemini TTS ➔ Kokoro TTS).
- **Enterprise Storage & Deployment**: Scalable AWS S3 media management, PostgreSQL persistent state, and zero-downtime GitHub Actions CI/CD to AWS EC2 with full HTTPS.

---

## 🏗️ End-to-End Architecture

```mermaid
flowchart TD
    subgraph Client ["Client Layer"]
        User["User Browser"] -->|HTTPS / Port 443| Domain["audioprep-ai.duckdns.org"]
        Domain -->|Nginx Ingress| Frontend["React 18 SPA (Vite)"]
    end

    subgraph AWS_EC2 ["AWS EC2 Host (c7i-flex.large / Sydney)"]
        Frontend -->|Reverse Proxy /api| Backend["FastAPI REST Server"]
        
        subgraph Pipeline ["Podcast Generation Pipeline"]
            direction TB
            Ingest["1. Ingestion: PyMuPDF + pymupdf4llm"]
            Chunk["2. Chunking: Structure-Aware / Semantic / Contextual / Recursive"]
            Index["3. Hybrid Indexing: Qdrant Dense + BM25 Sparse"]
            Planner["4. LangGraph Topic Planner (Concept Dependency)"]
            Dialogue["5. Dialogue Generation (Host <-> Expert)"]
            Critic{"6. Quality Critic: Groundedness & Relevance"}
            TTS["7. Audio Synthesis: Gemini TTS (Fallback: Kokoro)"]
            FFmpeg["8. Audio Polish: FFmpeg -16 LUFS Normalization"]
            
            Ingest --> Chunk --> Index --> Planner --> Dialogue --> Critic
            Critic --"Score < Threshold (Max 1 loop)"--> Dialogue
            Critic --"Score Approved"--> TTS --> FFmpeg
        end

        Backend --> Pipeline
        Backend <-->|SQLAlchemy 2.0 Async| Postgres[("PostgreSQL 16 DB\n(Docker Volume: postgres_data)")]
    end

    subgraph AWS_Cloud ["AWS Cloud Services"]
        Backend -->|boto3 API| S3_Bucket[("AWS S3 Bucket\naudioprep-storage-263824")]
        S3_Bucket -.->|Presigned Audio/PDF URL| User
    end

    classDef aws fill:#FF9900,stroke:#232F3E,stroke-width:2px,color:white;
    classDef comp fill:#2496ED,stroke:#1D63ED,stroke-width:2px,color:white;
    classDef model fill:#00C7B7,stroke:#008080,stroke-width:2px,color:white;
    
    class S3_Bucket,AWS_EC2 aws;
    class Backend,Frontend,Postgres comp;
    class Pipeline,TTS,Critic model;
```

---

## ✨ Key Features

- **Multi-Speaker Conversational Synthesis**: Generates realistic banter, interruptions, and natural questions between two distinct voices (Male host *"Puck"* & Female expert *"Kore"*).
- **4 Selectable Chunking Strategies**: Empirically evaluate and pick between Recursive, Semantic, Contextual, or Structure-Aware document segmentations.
- **RRF Hybrid Retrieval**: Combines sentence-transformer vector search (`all-MiniLM-L6-v2`) with lexical BM25 using server-side Reciprocal Rank Fusion.
- **Stateful Agentic LangGraph Pipeline**: Graph execution with conditional branching, cycle boundaries, and automated quality remediation.
- **Dual-Engine Audio Generation**: Primary conversational synthesis via Google Gemini Multi-Speaker API with automatic local fallback to Kokoro-82M TTS.
- **Podcast Studio Audio Mastering**: Post-processes output with FFmpeg to reach EBU R128 / -16 LUFS broadcast volume with smooth intro/outro pauses.
- **AWS S3 + PostgreSQL Architecture**: Source documents and finished media are streamed through presigned URLs, keeping the EC2 server stateless and lightweight.
- **Automated CI/CD**: Pushing to `main` builds multi-stage Docker images to GitHub Container Registry (GHCR) and performs rolling updates on AWS EC2.
- **Free Custom Domain with SSL**: Bound to `audioprep-ai.duckdns.org` with Let's Encrypt automated TLS certificate renewal.

---

## 🔬 Deep-Dive Component Workflows

### 1. Document Ingestion (PyMuPDF + OCR)
- **Module:** [`backend/app/services/ingestion/pdf_parser.py`](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/services/ingestion/pdf_parser.py)
- **Memory Footprint:** Uses only ~100MB RAM (unlike heavy alternatives such as Docling which require 2-4GB).
- **Processing:** Extracts Markdown representation preserving `# H1`, `## H2`, bullet points, and code blocks via `pymupdf4llm`. Scanned image pages automatically fallback to PyMuPDF's Tesseract-compatible textpage OCR pipeline.

### 2. Multi-Strategy Chunking Engine
- **Module:** [`backend/app/services/chunking/`](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/services/chunking/)
- Users can tailor chunking to document structure:

| Strategy | Mechanism | Best Suited For |
| :--- | :--- | :--- |
| **Structure-Aware** *(Default)* | Splits on markdown header hierarchies (`H1` ➔ `H2` ➔ `H3`), preserving logical sections. | Architecture docs, RFCs, technical guides |
| **Semantic** | Calculates cosine distance between consecutive sentences; splits when semantic variance spikes. | Dense academic papers, prose |
| **Contextual** | LLM prepends a 2-sentence document summary to each chunk to prevent orphaned context. | Fragmented FAQs, technical reference tables |
| **Recursive Character** | Hierarchical token split on `\n\n`, `\n`, `. `, and spaces. | General notes, unstructured dumps |

### 3. Hybrid Dense + Sparse Retrieval (Qdrant + BM25)
- **Module:** [`backend/app/services/retrieval/`](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/services/retrieval/)
- **Dense Vectors:** Encoded using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) into an in-memory/embedded Qdrant index. Captures conceptual semantics.
- **Sparse BM25:** Indexed via `rank-bm25` over tokenized words. Captures precise acronyms, function names, and technical terminology.
- **Reciprocal Rank Fusion (RRF):** Merges candidate rankings using standard score formula:
  $$\text{Score}(d) = \sum_{m \in \{\text{dense}, \text{sparse}\}} \frac{1}{60 + \text{rank}_m(d)}$$
- **Reranker:** Top candidates pass through a Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) scoring query-document pairs jointly to filter out noise.

### 4. LangGraph Agentic Orchestration
- **Module:** [`backend/app/agents/podcast_graph.py`](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/agents/podcast_graph.py)
- Orchestrated through LangGraph's cyclic `StateGraph`:
  1. **Topic Planner Node**: Breaks extracted document into a logical progression of concepts and interview question sequences.
  2. **Retrieval Node**: Gathers cross-referenced evidence chunks for each planned topic.
  3. **Dialogue Generation Node**: LLM crafts turns using strict speaker rules:
     - **HOST:** Asks clarifying questions, uses analogies, steers pacing.
     - **EXPERT:** Explains mechanisms, references citations, clarifies subtleties.
  4. **Quality Critic Node**: Inspects dialogue for hallucinations, coverage of key concepts, and natural flow. If score < 7.0, routes back to dialogue generation with targeted critique notes (bounded to 1 revision loop).

### 5. Dual-Engine TTS & Audio Engineering
- **Module:** [`backend/app/services/tts/`](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/services/tts/) & [`backend/app/services/audio/processor.py`](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/services/audio/processor.py)
- **Primary Engine:** Google Gemini Multi-Speaker API synthesizes natural two-person speech with emotional cues in a single multi-turn request.
- **Local Fallback:** If API limits or connectivity fails, routes gracefully to local **Kokoro-82M TTS** on CPU without failing the job.
- **Audio Post-Processing (FFmpeg):**
  - Loudness normalization to **-16 LUFS** (podcast international broadcast standard).
  - Silence trimming & intelligent conversational gap insertion (0.4s natural speaker transition).
  - Encoded to 192kbps stereo MP3.

### 6. PostgreSQL State & AWS S3 Object Storage
- **Modules:** [`backend/app/core/database.py`](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/core/database.py) & [`backend/app/services/storage/s3_service.py`](file:///c:/Users/athar/Downloads/the-podcaster/backend/app/services/storage/s3_service.py)
- **Database (`jobs` table):**
  - Tracks UUID `job_id`, real-time stage progress (0–100%), error diagnostics, and timestamps.
  - Retains S3 object keys and generated presigned streaming URLs.
- **AWS S3 (`audioprep-storage-263824`):**
  - Source documents: `pdfs/{job_id}/{filename}`
  - Output podcasts: `media/{job_id}/podcast_output.mp3`
  - High durability; ensures the host server's local disk remains unburdened.

---

## 🌐 Cloud Deployment & Networking (AWS + DuckDNS)

### Why DuckDNS & Dynamic DNS Explained
In enterprise cloud setups, binding a bare AWS Public IP address (`http://13.236.98.48`) directly to end users creates critical issues:
1. **No Free SSL on Raw IPs:** Public Certificate Authorities (like Let's Encrypt) do not issue free SSL/TLS certificates for bare IP addresses; they require a fully qualified domain name (FQDN).
2. **Browser Blocking:** Modern desktop and mobile browsers (Chrome 117+, Safari, Edge) automatically force `https://` on URL entries or display severe `ERR_CONNECTION_REFUSED` / "Dangerous Site" warnings when accessing HTTP over raw IPs.
3. **Dynamic IP Changes:** If an AWS EC2 instance is stopped and restarted without an Elastic IP, its public IP changes, breaking all bookmarks.

**The DuckDNS Solution:**
- **DuckDNS** provides a free, lightweight Dynamic Domain Name Service (DDNS).
- By registering `audioprep-ai.duckdns.org` and mapping it to the EC2 Public IPv4 (`13.236.98.48`), the server gains an authoritative FQDN at **$0.00 cost**.
- Allows automated DNS resolution so any user or recruiter clicking the link is routed directly to the AWS EC2 instance.

### Let's Encrypt Automated SSL/TLS (Certbot)
With `audioprep-ai.duckdns.org` pointed at the server, **Certbot** was executed via standalone ACME challenge:
```bash
sudo certbot certonly --standalone -d audioprep-ai.duckdns.org --agree-tos --register-unsafely-without-email
```
1. Certbot validates domain ownership against DuckDNS.
2. It generates trusted cryptographic certificates stored at `/etc/letsencrypt/live/audioprep-ai.duckdns.org/`.
3. The Docker container mounts `/etc/letsencrypt` as a read-only volume.
4. **Nginx** handles TLS termination:
   - Port 80 requests receive an instant `301 Moved Permanently` redirect to `https://`.
   - Port 443 serves TLS 1.2/1.3 with modern cipher suites and terminates the secure connection before proxying to FastAPI.

### AWS Infrastructure (Cost-Engineered)
- **Region:** `ap-southeast-2` (Asia Pacific - Sydney)
- **Compute:** EC2 **`c7i-flex.large`** (4th Gen Intel Xeon Scalable, 2 vCPUs, 4 GiB RAM)
  - Selected for dedicated compute performance over burstable `t3` instances, providing fast vector embeddings and FFmpeg audio mastering.
- **Storage:** 30 GB gp3 SSD (EC2 root) + AWS S3 Standard (Object storage).
- **Security Groups:**
  - `Port 22` (SSH Management)
  - `Port 80` (HTTP Ingress ➔ Redirects to HTTPS)
  - `Port 443` (HTTPS Ingress ➔ Nginx TLS Termination)

---

## 🔄 CI/CD Pipeline (GitHub Actions)

Every push to the `main` branch triggers an automated build, test, package, and deploy cycle defined in [`.github/workflows/deploy.yml`](file:///c:/Users/athar/Downloads/the-podcaster/.github/workflows/deploy.yml):

```
git push origin main
       │
       ▼
┌────────────────────────────────────────┐
│  GitHub Actions Runner (Ubuntu)        │
│  ├── Checkout Repository               │
│  ├── Docker Buildx Setup               │
│  ├── Build Backend Docker Image        │
│  ├── Build Frontend React Docker Image │
│  └── Push to GHCR (ghcr.io)            │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  Continuous Deployment (SSH to EC2)    │
│  ├── Pre-pull cleanup: docker prune    │
│  ├── Pull latest images from GHCR      │
│  ├── Recreate containers via Compose   │
│  └── Post-cleanup: prune stale layers  │
└────────────────────────────────────────┘
                   │
                   ▼
         Live at audioprep-ai.duckdns.org
```

### Self-Cleaning Disk Protection
Because machine learning containers containing PyTorch and audio binaries are ~3.5 GB, deploying multiple revisions can fill server storage. The deployment script runs automated image pruning:
```bash
docker image prune -af || true
docker compose pull backend frontend
docker compose up -d --remove-orphans
docker image prune -af || true
```
This guarantees that stale images are scrubbed on every release while keeping the PostgreSQL database volume (`postgres_data`) and S3 objects completely untouched.

---

## 📁 Project Directory Structure

```text
the-podcaster/
├── .github/
│   └── workflows/
│       └── deploy.yml             # GitHub Actions CI/CD deployment pipeline
├── backend/
│   ├── app/
│   │   ├── agents/                # LangGraph state machine & agent nodes
│   │   │   ├── podcast_graph.py   # StateGraph definition & execution logic
│   │   │   ├── topic_planner.py   # Concept dependency curriculum planner
│   │   │   ├── quality_critic.py  # Factuality & dialogue critic
│   │   │   └── state.py           # Pydantic agent state definitions
│   │   ├── core/                  # Core configurations and infrastructure
│   │   │   ├── config.py          # Pydantic Settings (Environment variables)
│   │   │   ├── database.py        # SQLAlchemy 2.0 Async PostgreSQL engine
│   │   │   ├── job_store.py       # Hybrid memory + PostgreSQL persistent store
│   │   │   └── llm_provider.py    # Fallback provider: NVIDIA -> Gemini -> Groq
│   │   ├── models/                # Schemas & ORM database models
│   │   │   ├── db_models.py       # SQLAlchemy JobEntity table definition
│   │   │   └── job.py             # Pydantic REST schemas for API requests
│   │   ├── services/              # Domain processing services
│   │   │   ├── audio/processor.py # FFmpeg normalization & mixing
│   │   │   ├── chunking/          # 4 chunking strategy implementations
│   │   │   ├── dialogue/          # Prompt templates & dialogue generation
│   │   │   ├── ingestion/         # PyMuPDF document extraction & OCR
│   │   │   ├── retrieval/         # Qdrant Dense + BM25 Sparse hybrid search
│   │   │   ├── storage/s3_service # AWS S3 client & presigned URL generator
│   │   │   └── tts/               # Gemini Multi-Speaker & Kokoro fallback
│   │   └── main.py                # FastAPI HTTP routing & background tasks
│   ├── Dockerfile                 # Multi-stage Python 3.11 + FFmpeg container
│   └── requirements.txt           # Python dependencies
├── frontend/
│   ├── src/
│   │   ├── components/            # React UI components
│   │   │   ├── AudioPlayer.jsx    # Custom audio playback & seeking player
│   │   │   ├── FileUpload.jsx     # Drag-and-drop PDF upload component
│   │   │   ├── PodcastConfig.jsx  # Duration, difficulty & strategy config
│   │   │   └── StatusTracker.jsx  # Real-time pipeline stage progress bar
│   │   ├── App.jsx                # Main application orchestrator
│   │   └── index.css              # Custom styling
│   ├── Dockerfile                 # Multi-stage Node builder + Nginx image
│   ├── nginx.conf                 # Production Nginx reverse proxy + SSL config
│   └── package.json               # Node.js dependencies
├── docker-compose.yml             # Orchestration for Postgres, Backend & Frontend
├── .gitignore                     # Git exclusion rules for API keys & binaries
└── README.md                      # Comprehensive project documentation
```

---

## 💻 Local Development Setup

### Prerequisites
- Python 3.11+
- Node.js 20+ & npm
- FFmpeg installed locally (for audio mastering)
- Docker & Docker Compose (optional, for containerized run)

### 1. Clone the Repository
```bash
git clone https://github.com/12ATHARAV/AudioPrep-AI.git
cd AudioPrep-AI
```

### 2. Configure Environment Variables
Create `backend/.env` based on the template:
```env
# LLM Providers (At least one required)
GEMINI_API_KEY=your_gemini_api_key
NVIDIA_API_KEY=your_nvidia_api_key
GROQ_API_KEY=your_groq_api_key
LLM_PRIMARY=nvidia

# AWS S3 Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=ap-southeast-2
AWS_S3_BUCKET=your_s3_bucket_name

# Database Configuration (Defaults to local PostgreSQL or SQLite fallback)
DATABASE_URL=postgresql+asyncpg://audioprep:audioprep_secure_pass@localhost:5432/audioprep_db
```

### 3. Run with Docker Compose (Recommended)
Launch the entire system (PostgreSQL + FastAPI + React + Nginx) with a single command:
```bash
docker compose up --build
```
- Open `http://localhost:80` for the Web App.
- Open `http://localhost:8000/docs` for the interactive API Swagger Documentation.

### 4. Or Run Services Manually

**Start Backend:**
```bash
cd backend
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Start Frontend:**
```bash
cd frontend
npm install
npm run dev
```
Access the development UI at `http://localhost:5173`.

---

## 📡 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/api/podcasts` | Uploads PDF, pushes to S3, creates DB record, and triggers async generation. |
| `GET` | `/api/podcasts/{job_id}` | Polls job progress, pipeline stages, and receives S3 streaming audio URLs. |
| `GET` | `/api/podcasts/{job_id}/audio` | Direct audio download endpoint (fallback if direct S3 stream is bypass-requested). |
| `GET` | `/api/health` | Healthcheck endpoint reporting active LLM and TTS status. |
| `GET` | `/docs` | OpenAPI / Swagger interactive documentation UI. |

---

## 📊 RAGAS Evaluation Framework

To ensure audio transcripts remain faithful to source materials, AudioPrep AI implements automated RAG evaluation using **RAGAS** ([`backend/evals/`](file:///c:/Users/athar/Downloads/the-podcaster/backend/evals/)):

1. **Context Precision**: Evaluates whether retrieved PDF chunks strictly correspond to topic concepts.
2. **Context Recall**: Verifies all critical technical dependencies were identified by the Topic Planner.
3. **Faithfulness**: Measures whether claims made by the Host and Expert are factually grounded in the source text.
4. **Answer Relevancy**: Assesses whether dialogue explanations address technical concepts clearly and concisely.

Benchmark script:
```bash
python backend/evals/run_eval.py
```

---

## 📄 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

<div align="center">
  <b>AudioPrep AI</b> — Transform technical notes into your commute's best podcast.<br>
  Built with ❤️ by <a href="https://github.com/12ATHARAV">12ATHARAV</a>
</div>
