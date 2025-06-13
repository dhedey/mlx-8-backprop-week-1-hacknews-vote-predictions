# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the training script and other necessary files
COPY train.py .
COPY .env .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the training script
CMD ["python", "train.py"] 