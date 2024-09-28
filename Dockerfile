# Use an official Python runtime as a base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only specific files (requirements.txt, app.py, and model.pth)
COPY requirements.txt /app/
COPY app.py /app/
COPY best_model.pth /app/

# Install dependencies from requirements.txt
RUN pip install -r requirements.txt
RUN mkdir -p /app/logs

# Expose port 8000 for the FastAPI app
EXPOSE 8000

# Set environment variable
ENV API_KEY=ABDELHAMEMELOPSE

# Command to run the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]

#opentelemetry-instrument \
 #   --traces_exporter console,otlp \
 #   --metrics_exporter console \
 #  --service_name your-service-name \
 #   --exporter_otlp_endpoint 0.0.0.0:4317 \
#   python myapp.py
