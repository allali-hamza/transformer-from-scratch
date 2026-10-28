FROM python:3.12-slim

WORKDIR /app

ENV HF_HOME=/opt/huggingface

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY predict.py ./

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='hamza-allali/linux-command-transformer')"

ENTRYPOINT ["python", "predict.py"]

