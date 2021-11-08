# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace --gpus all -it --rm <project_name>


FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Basic setup
RUN apt update
RUN apt install -y bash \
                   htop \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   wget \
                   && rm -rf /var/lib/apt/lists

# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt
