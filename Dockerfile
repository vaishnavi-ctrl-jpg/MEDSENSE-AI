# 1. Start with a lightweight Python base
FROM python:3.10-slim

# 2. Set the folder where everything will happen inside the container
WORKDIR /app

# 3. Copy only the requirements first (this makes builds faster!)
COPY requirements.txt .

# 4. Install all the libraries Vivi and you need
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy every single file from your project into the container
COPY . .

# 6. Open the port that FastAPI uses
EXPOSE 7860

# 7. The command to start your FastAPI app when the container runs
CMD ["uvicorn", "server.openenv_api:app", "--host", "0.0.0.0", "--port", "8000"]