FROM python:3.10-slim

WORKDIR /usr/src/app

# Copy local code to the container image
COPY . ./

# Install system dependencies required for dlib and other libraries
RUN apt-get update && apt-get install -y cmake g++ make libopencv-dev

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 5000

# Start the Gunicorn server with the appropriate binding to $PORT
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]
