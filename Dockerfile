# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for Tesseract OCR and OpenCV (used by Pillow/Transformers)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    # Add language data for Tesseract - e.g., English. Add others if needed.
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install Python dependencies
# Ensure your requirements.txt specifies torch for CPU if you don't need GPU on App Service
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container at /app
COPY . .

# Make port 8000 available to the world outside this container
# Azure App Service will route external port 80/443 to this port
EXPOSE 8000

# Define the command to run your application
# Using Gunicorn with Uvicorn workers is recommended for production
# Ensure app.main:app matches your FastAPI application instance
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "app.main:app"]