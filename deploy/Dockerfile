# 1. use a lightweight python linux image
FROM python:3.9-slim

# 2. set up the working directory inside the container
WORKDIR /app

# 3. copy dependencies first (caching trick to make builds faster)
COPY requirements.txt .

# 4. install the libraries
RUN pip install --no-cache-dir -r requirements.txt

# 5. copy your actual code
# we copy specific folders to keep the container clean
COPY src/ ./src/
COPY deploy/ ./deploy/
COPY saved_models/ ./saved_models/

# 6. open the port so the world can talk to your api
EXPOSE 8000

# 7. the command to start the api server
CMD ["uvicorn", "deploy.app:app", "--host", "0.0.0.0", "--port", "8000"]
