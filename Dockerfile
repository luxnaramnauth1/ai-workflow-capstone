FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run the full unit-test suite at build time: the image only builds if all tests pass
RUN python run-tests.py

EXPOSE 8080
CMD ["python", "app.py"]
