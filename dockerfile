# Use Python 3.11.9 as a base image
FROM python:3.11.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements files to the container
COPY requirements.txt /app/

# Install development dependencies
RUN python -m pip install --upgrade -r requirements.txt

# Copy the backend folder into the container
COPY backend/ /app/backend/

# Set the working directory to the backend folder
WORKDIR /app/backend

# Make port 80 available to the world outside this container
EXPOSE 80

# Run your application
CMD ["python", "appfinal.py"]
