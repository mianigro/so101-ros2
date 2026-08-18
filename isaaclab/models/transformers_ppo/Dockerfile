FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create anon user
RUN groupadd -g 1000 anon && \
    useradd -u 1000 -g anon -s /bin/bash anon

# Set ownership of /app directory
RUN chown -R anon:anon /app

# Copy requirements
COPY --chown=anon:anon requirements.txt /app/

# Install requirements and then mamba-ssm
RUN pip3 install -r requirements.txt
RUN pip3 install mamba-ssm --no-build-isolation

# Copy project files
COPY --chown=anon:anon . /app/

# Switch user
USER anon

# TensorBoard port
EXPOSE 6006

# Default command
CMD ["bash"]