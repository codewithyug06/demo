# ==============================================================================
# SENTINEL PRIME V9.8 | SOVEREIGN CONTAINER PROTOCOL
# ENTERPRISE-GRADE DOCKERFILE FOR HIGH-SECURITY DEPLOYMENT
# COMPLIANCE: UIDAI Data Security Guidelines 2026
# ==============================================================================

# 1. BASE IMAGE SELECTION
# Using Python 3.9 Slim to balance image size with scientific library compatibility
FROM python:3.9-slim as base

# 2. METADATA LABELS (GOVERNANCE)
LABEL maintainer="Sentinel Prime Team <admin@sentinel-prime.gov>"
LABEL version="9.8.0-AEGIS"
LABEL description="Sovereign Digital Twin Analytics Engine for Aadhaar Hackathon 2026"
LABEL security.compliance="Zero-Trust/Local-Compute"

# 3. ENVIRONMENT CONFIGURATION
# Prevent Python from writing pyc files to disc (Security & Size)
ENV PYTHONDONTWRITEBYTECODE=1
# Ensure stdout/stderr are flushed immediately (Real-time Logging)
ENV PYTHONUNBUFFERED=1
# Set Locale settings
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
# Hardcode Timezone to IST (Indian Standard Time) for UIDAI Audit Accuracy
ENV TZ=Asia/Kolkata

# Sentinel Config Defaults (Override these in docker run)
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV OPENAI_API_KEY=""
ENV MAPBOX_TOKEN=""

# 4. SYSTEM DEPENDENCY LAYER (THE HEAVY LIFTING)
# Install build essentials for compiling PyTorch/SciPy/NumPy extensions
# Install GDAL/GEOS for Advanced Geospatial Engine (Spatial.py)
# Install Fonts for FPDF Executive Brief generation (Cognitive.py)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    make \
    curl \
    git \
    libgdal-dev \
    libatlas-base-dev \
    gfortran \
    fonts-liberation \
    fontconfig \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 5. WORKSPACE SETUP
WORKDIR /app

# 6. PYTHON DEPENDENCY LAYER (OPTIMIZED CACHING)
# Copy requirements first. Docker will cache this layer if requirements.txt doesn't change.
COPY requirements.txt .

# Upgrade pip and install dependencies with extended timeout for large ML wheels
# Installs: torch, pandas, numpy, scikit-learn, networkx, pydeck, streamlit, fpdf, h3
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Install production server utilities explicitly
RUN pip install --no-cache-dir streamlit gunicorn uvicorn

# 7. SOURCE CODE INGESTION
# Copy the entire project structure into the container
COPY . .

# 8. SECURITY HARDENING: NON-ROOT USER
# Create a specialized user 'sentinel' to run the application
# This prevents potential privilege escalation attacks (Zero-Trust)
RUN groupadd -r sentinel && useradd -r -g sentinel sentinel

# Transfer ownership of the application directory to the non-root user
RUN chown -R sentinel:sentinel /app

# Switch context to non-root user
USER sentinel

# 9. HEALTHCHECK PROTOCOL
# Periodically pings the Streamlit health endpoint to ensure the Aegis Command is active
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 10. NETWORK EXPOSURE
# Expose Streamlit Dashboard Port
EXPOSE 8501
# Expose API Port (Reserved for future microservices)
EXPOSE 8000

# 11. EXECUTION ENTRYPOINT
# By default, runs the Main Analysis Pipeline (God Mode Output)
# To run the Dashboard, override this CMD with: ["streamlit", "run", "interface/dashboard.py"]
CMD ["python", "main.py"]