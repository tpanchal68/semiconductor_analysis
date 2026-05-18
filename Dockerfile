# --- Stage 1: Base/Dev (Includes everything) ---
# To run tests: docker build --target base -t sauca-test .
FROM python:3.12.10-slim AS base
WORKDIR /Sauca_AI
# 1. Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libboost-all-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. Proceed with Python dependencies
COPY requirements.txt .
COPY requirements_test.txt .
RUN pip install --no-cache-dir -r requirements.txt -r requirements_test.txt
COPY . .

# --- Stage 2: Production (Clean & Small) ---
# To run for real: docker build --target production -t sauca-prod .
FROM python:3.12.10-slim AS production
WORKDIR /Sauca_AI
# 1. Install system-level dependencies
# Install ONLY the small runtime libraries needed to RUN the code
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libboost-python1.74.0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled python packages from the base stage
COPY --from=base /root/.local /root/.local
# 2. Proceed with Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Ensure the app can find the copied packages
ENV PATH=/root/.local/bin:$PATH
EXPOSE 4700
CMD ["python", "run.py"]
