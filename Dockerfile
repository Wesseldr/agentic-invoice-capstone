# Using a small and light Python image
FROM python:3.11-slim

# 1. Install system dependency (Poppler for PDF processing is crucial)
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# 2. Set working dir
WORKDIR /app

# 3. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy all code & data files
COPY . .

# 5. Make executable
RUN chmod +x demo.sh

# 6. Direct output (logging)
ENV PYTHONUNBUFFERED=1

# 7. Run full demo
CMD ["./demo.sh"]
