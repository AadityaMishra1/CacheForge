FROM --platform=linux/amd64 condaforge/miniforge3:24.11.3-0

# Default name for the project conda environment.
ARG CONDA_ENV_NAME=cacheforge
ARG MAMBA_USER=mambauser

# Provide the shared package cache path expected by environment.yml.
ENV GROUP=cacheforge
RUN mkdir -p /share/${GROUP}/${MAMBA_USER}/conda/pkgs

# System dependencies needed for compiling ChampSim policies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

RUN id -u ${MAMBA_USER} >/dev/null 2>&1 || useradd -m ${MAMBA_USER}

# Copy project files into the image layout and ensure permissions match the mamba user.
WORKDIR /workspace
COPY . /workspace
RUN chown -R ${MAMBA_USER}:${MAMBA_USER} /workspace

# Switch to the non-root mamba user for environment creation and runtime.
USER ${MAMBA_USER}
ENV USER=${MAMBA_USER}
SHELL ["/bin/bash", "-lc"]

# Create the conda environment and activate it for interactive shells.
RUN mamba env create -y -n ${CONDA_ENV_NAME} -f environment.yml
RUN echo "conda activate ${CONDA_ENV_NAME}" >> ~/.bashrc

# Expose the environment binary path for non-interactive commands.
ENV PATH=/opt/conda/envs/${CONDA_ENV_NAME}/bin:${PATH}

CMD ["bash"]
