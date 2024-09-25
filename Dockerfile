FROM python:3.9.17-bookworm

# Allow statements and log messages to immediately appear in the logs
ENV PYTHONUNBUFFERED True
ENV APP_HOME /back-end
WORKDIR $APP_HOME

# Copy local code to the container image
COPY . ./

# Install system dependencies required for dlib and other libraries
RUN apt-get update && apt-get install -y cmake g++ make libopencv-dev

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
ENV PORT 8000

# Start the Gunicorn server with the appropriate binding to $PORT
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
