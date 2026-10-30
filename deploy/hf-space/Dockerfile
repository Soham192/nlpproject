# Backend for the redaction demo, built as a Hugging Face Docker Space (free CPU tier: 2 vCPU, 16 GB RAM).
# Assembled and pushed by deploy/push_hf_space.py; the build context is the Space repo root.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000; the app dir must be theirs (uploads, outputs and key.bin live there).
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    NLTK_DATA=/home/user/nltk_data \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /home/user/app

# CPU-only torch first (the CUDA wheels would add GBs for nothing), then everything else pinned against it.
COPY --chown=user requirements-space.txt constraints-space.txt ./
RUN pip install --user --index-url https://download.pytorch.org/whl/cpu \
        torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    && pip install --user -c constraints-space.txt -r requirements-space.txt

# Bake every model into the image so a Space waking from sleep never downloads anything.
COPY --chown=user config.yaml prefetch.py ./
RUN python prefetch.py

COPY --chown=user src ./src
COPY --chown=user api ./api
COPY --chown=user start.sh ./
RUN chmod +x start.sh

EXPOSE 7860
CMD ["./start.sh"]
