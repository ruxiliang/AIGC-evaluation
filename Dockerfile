FROM nvidia/cuda:11.8.0-devel-ubuntu20.04
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
ARG DEBIAN_FRONTEND=noninteractive


RUN apt-get install -y wget git libglib2.0-0 libsm6 libxrender1 libxext6 tmux cuda-nsight-systems-11-8 nsight-systems-2022.4.2 \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

SHELL [ "/bin/bash", "-c" ]

RUN git clone https://github.com/CompVis/stable-diffusion \
    && cd /stable-diffusion \ 
    && echo ${PATH} \
    && source activate \
    && conda env create -f environment.yaml \
    && conda activate ldm \
    && conda config --set always_yes yes --set changeps1 no \
    && conda install pytorch torchvision -c pytorch \
    && pip install transformers==4.19.2 diffusers==0.12.1 invisible-watermark \
    && pip install -e .\
    && pip install pynvml \
    && cd .. \
    && rm -rf /stable-dffusion


RUN sed -i '5s/PILLOW_VERSION/__version__ as PILLOW_VERSION/' /root/miniconda3/envs/ldm/lib/python3.8/site-packages/torchvision/transforms/functional.py

RUN source activate ldm && pip install clip taming-transformers-rom1504 
# RUN source activate && conda activate ldm && python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms && rm -rf outputs

# RUN source activate && conda activate ldm && mkdir -p ghexample && wget --output-document ghexample/example.jpg https://github.com/CompVis/stable-diffusion/blob/main/assets/stable-samples/img3img/sketch-mountains-input.jpg && python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img ghexample/example.jpg --strength 0.8 && rm -rf outputs
