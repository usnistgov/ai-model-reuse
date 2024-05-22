
FROM continuumio/miniconda3
MAINTAINER National Institute of Standards and Technology

# Define exec and data folders
ARG EXEC_DIR="/opt/executables"
ARG DATA_DIR="/data"

# Create folders
RUN mkdir -p ${EXEC_DIR} \
    && mkdir -p ${DATA_DIR}/inputs \
    && mkdir -p ${DATA_DIR}/outputs

# Copy code and custom settings file
RUN cd ..
COPY ./. ${EXEC_DIR}/

# Set workdir
WORKDIR ${EXEC_DIR}

# Create environment
RUN conda create -n airec_test python=3.8

RUN echo "source activate airec_test" > ~/.bashrc
ENV PATH /opt/conda/envs/airec_test/bin:$PATH
#RUN conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch -c conda-forge
#RUN pip install numpy==1.19.2 Cython==3.0a1
#RUN pip install -U scikit-learn kaleido
#RUN pip install scikit-image GPUtil tqdm lmdb protobuf pandas matplotlib plotly
#RUN ls -la

RUN pip install -r requirements.txt
#RUN conda install
ENTRYPOINT ["./WIPP/entrypoint.sh"]