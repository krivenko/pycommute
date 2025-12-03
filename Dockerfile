FROM ubuntu:24.04 AS base
LABEL maintainer="Igor Krivenko <iskrivenko@proton.me>"
LABEL description="libcommute/pycommute demonstration image"

# libcommute Git branch/tag name
ARG LIBCOMMUTE_GIT_REF_NAME=master

# Suppress all confirmation dialogs from apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Create docker user
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN useradd -m -s /bin/bash -u 9999 docker && echo "docker:docker" | chpasswd

# Install dependencies
# hadolint ignore=DL3008
RUN <<EOT
apt-get update
apt-get install -y --no-install-recommends \
    git build-essential coreutils cmake \
    libeigen3-dev libboost-dev libgmp-dev \
    python3 libpython3-dev python3-venv
apt-get clean
rm -rf /var/lib/apt/lists/*
EOT

#
# Build libcommute & pycommute
#

FROM base AS builder
USER docker
WORKDIR /home/docker

# Clone libcommute sources
RUN git clone --branch "${LIBCOMMUTE_GIT_REF_NAME}" \
        https://github.com/krivenko/libcommute libcommute.git

# Configure libcommute build
WORKDIR /home/docker/libcommute.build
RUN cmake ../libcommute.git -DCMAKE_INSTALL_PREFIX=/home/docker/libcommute \
          -DCMAKE_BUILD_TYPE=Release \
          -DTESTS=ON \
          -DEXAMPLES=ON

# Build, test and install libcommute
RUN make -j4 VERBOSE=1 && ctest --output-on-failure && make install

# Create and activate virtual environment
RUN python3 -m venv /home/docker/venv
ENV PATH="/home/docker/venv/bin:$PATH"

# Copy pycommute sources into the builder
COPY --chown=docker:docker . /home/docker/pycommute

# Install Python dependencies
WORKDIR /home/docker/pycommute
RUN <<EOT
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir scipy==1.16.* jupyter==1.1.* jupytext==1.18.*
EOT

# Build and install pycommute
ENV LIBCOMMUTE_INCLUDEDIR=/home/docker/libcommute/include
RUN pip install --no-cache-dir --verbose .

# Convert pycommute examples to Jupyter notebooks
WORKDIR /home/docker/pycommute/docs/examples
RUN jupytext --to notebook --set-kernel python3 -- *.py && rm -- *.py

#
# Create application image with Jupyter setup
#

FROM base AS app
USER docker
WORKDIR /home/docker
ENV PATH="/home/docker/venv/bin:$PATH"

# Copy files from the builder to the app image
COPY --from=builder --chown=docker:docker /home/docker/venv venv
COPY --from=builder --chown=docker:docker /home/docker/pycommute/docs/examples \
     pycommute_examples

# Run Jupyter notebook
WORKDIR /home/docker/pycommute_examples
EXPOSE 8888
CMD ["jupyter","notebook","--ip","0.0.0.0"]
