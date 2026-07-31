FROM dustynv/pytorch:2.7-r36.4.0

# ── NVIDIA runtime ────────────────────────────────────────────────────────────
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# ── System tools (gh CLI + wget, same as v1) ──────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends wget \
    && mkdir -p -m 755 /etc/apt/keyrings \
    && wget -nv -O /etc/apt/keyrings/githubcli-archive-keyring.gpg \
       https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update && apt-get install -y gh \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
# torch / torchvision are already in the dustynv base — skip them here
RUN pip3 install --no-cache-dir \
    --index-url https://pypi.org/simple/ \
    jupyter \
    "grpcio>=1.60.0" \
    "grpcio-tools>=1.60.0" \
    "protobuf>=4.25" \
    "matplotlib>=3.7" \
    "numpy>=1.24"

# Disable Jupyter auth — wipe any pre-existing config from the base image first
RUN rm -rf /root/.jupyter && mkdir -p /root/.jupyter && \
    jupyter notebook --generate-config -y && \
    printf '\nc.NotebookApp.token = ""\nc.NotebookApp.password = ""\n' \
      >> /root/.jupyter/jupyter_notebook_config.py && \
    jupyter server --generate-config -y && \
    printf '\nc.IdentityProvider.token = ""\nc.PasswordIdentityProvider.hashed_password = ""\nc.ServerApp.password = ""\n' \
      >> /root/.jupyter/jupyter_server_config.py

# ── Application code ──────────────────────────────────────────────────────────
WORKDIR /app/sutdDFL/ocdFL
COPY . .

# Pre-compile gRPC stubs so the image is self-contained
RUN bash compile_protos.sh

EXPOSE 8888

# Default: run DFL training.
# Override for Jupyter: docker run ... jupyter notebook --ip=0.0.0.0 --allow-root --no-browser --NotebookApp.token=''
CMD ["bash", "run.sh"]
