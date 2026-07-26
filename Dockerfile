# 1. Use official lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy your requirement file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of your application code
COPY iris_fastapi.py .
COPY download_model.py .

# 5. Define build arguments for MLflow credentials (passed in by GitHub Actions later)
ARG MLFLOW_TRACKING_URI
ARG MLFLOW_TRACKING_USERNAME
ARG MLFLOW_TRACKING_PASSWORD

# Set them as environment variables so the Python script can read them
ENV MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
ENV MLFLOW_TRACKING_USERNAME=$MLFLOW_TRACKING_USERNAME
ENV MLFLOW_TRACKING_PASSWORD=$MLFLOW_TRACKING_PASSWORD

# 6. Execute the script to fetch the model from MLflow and bundle it into the image
RUN python download_model.py

# 7. Expose the port the app runs on
EXPOSE 8200

# 8. Command to start the FastAPI server
CMD ["uvicorn", "iris_fastapi:app", "--host", "0.0.0.0", "--port", "8200"]