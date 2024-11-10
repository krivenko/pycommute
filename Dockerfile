FROM ubuntu:20.04 as base
LABEL maintainer="Igor Krivenko <iskrivenko@proton.me>"
LABEL description="libcommute/pycommute demonstration image"

# Suppress all confirmation dialogs from apt-get
ENV DEBIAN_FRONTEND noninteractive

# Create docker user
RUN useradd -m -s /bin/bash -u 999 docker && echo "docker:docker" | chpasswd

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    coreutils \
    cmake \
    python3 \
    libpython3-dev \
    python3-setuptools \
    python3-wheel \
    python3-pip

FROM base as builder
USER docker
WORKDIR /home/docker

# Build and install libcommute
RUN git clone https://github.com/krivenko/libcommute libcommute.git && \
    mkdir libcommute.build && \
    cd libcommute.build && \
    cmake ../libcommute.git -DCMAKE_INSTALL_PREFIX=${HOME}/.local \
          -DCMAKE_BUILD_TYPE=Release \
          -DTESTS=ON \
          -DEXAMPLES=ON && \
    make -j3 && \
    make test && \
    make install

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN export PATH=${HOME}/.local/bin:${PATH} && \
    pip3 install --user -r requirements.txt && \
    pip3 install --user scipy jupyter p2j

# Build and install pycommute
COPY --chown=docker:docker . pycommute
WORKDIR /home/docker/pycommute
RUN LIBCOMMUTE_INCLUDEDIR=/home/docker/.local/include \
    python3 setup.py install --user

# Prepare and run Jupyter
FROM base as app
USER docker
WORKDIR /home/docker

COPY --from=builder --chown=docker:docker /home/docker/.local .local
COPY --from=builder --chown=docker:docker /home/docker/pycommute/docs/examples \
     pycommute_examples
# Convert examples to Jupyter notebooks
RUN cd pycommute_examples && \
    for f in *.py; do /home/docker/.local/bin/p2j ${f}; done && \
    rm *.py

# Run Jupyter notebook
WORKDIR /home/docker/pycommute_examples
EXPOSE 8888
CMD ["/home/docker/.local/bin/jupyter","notebook","--ip","0.0.0.0"]
