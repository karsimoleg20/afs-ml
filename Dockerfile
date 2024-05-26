FROM ultralytics/ultralytics:latest-cpu

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "api.py"]
