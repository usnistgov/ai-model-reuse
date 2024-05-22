
#FROM nvidia/cuda:11.7.1-base-ubuntu22.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04
# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

RUN  apt-get update \
  && apt-get install -y wget \
  && rm -rf /var/lib/apt/lists/*
# Install some basic utilities.
#RUN apt-get update && apt-get install -y \
#    curl \
#    ca-certificates \
#    sudo \
#    git \
#    bzip2 \
#    libx11-6 \
# && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && bash /miniconda.sh -b -p /miniconda && rm /miniconda.sh

# Set path to conda
ENV PATH=/miniconda/bin:${PATH}
#RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
# && chown -R user:user /app
#RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
#USER user

MAINTAINER National Institute of Standards and Technology
# Define exec and data folders


# Create folders
RUN mkdir -p $"EXEC_DIR" && mkdir -p $"DATA_DIR"/inputs && mkdir -p $"DATA_DIR"/outputs

ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

# Copy code and custom settings file
RUN cd ..
COPY ./. ${EXEC_DIR}/

# Set workdir
WORKDIR ${EXEC_DIR}
# Install PyTorch 1.13.1 with CUDA 11.7 support
#RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 \
#    -f https://download.pytorch.org/whl/cu117/torch_stable.html
# Create environment
RUN conda create -n airec_test python=3.8.10 --channel default --channel pytorch --channel nvidia
ENV PATH /opt/conda/envs/airec_test/bin:$PATH
 #--extra-index-url https://download.pytorch.org/whl/cu117
RUN echo "source activate airec_test" > ~/.bashrc
RUN echo "nvidia-smi"
#RUN . $airec_test/bin/activate
SHELL ["conda", "run","-n","airec_test", "/bin/bash", "-c"]
RUN pip install numpy==1.19.2 Cython==3.0a1
RUN pip install -U scikit-learn kaleido
RUN pip install scikit-image GPUtil tqdm lmdb protobuf pandas matplotlib==3.6.1 plotly
RUN pip install seaborn imagecodecs tifffile
RUN pip install statannotations==0.6.0
RUN pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install torchsummary==1.5.1 torchsummaryX==1.3.0
#RUN pip install -r requirements.txt
#RUN conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -c conda-forge
#RUN ls -la
#RUN conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
RUN conda init bash
ENTRYPOINT ["./WIPP/entrypoint.sh"]