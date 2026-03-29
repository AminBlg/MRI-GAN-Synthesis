FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "train.py"]
CMD ["--dataroot", "Dataset", "--epochs", "10", "--output_dir", "output"]
