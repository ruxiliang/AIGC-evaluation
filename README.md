# Generative AI workload benchmarking and analysis job Steps
1. Create a folder that contains all of your AIGC jobs and set the environment variable as follows: 
```bash
export AIGC_DIR=/path/to/your/aigc-jobs
```
2. Build a docker container
	First use the dockerfile provided below and save it as `Dockerfile` in `$AIGC_DIR`:
```Dockerfile
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
    && pip install transformers diffusers==0.12.1 invisible-watermark \
    && pip install -e .\
    && pip install pynvml \
    && cd .. \
    && rm -rf /stable-dffusion




RUN sed -i '5s/PILLOW_VERSION/__version__ as PILLOW_VERSION/' /root/miniconda3/envs/ldm/lib/python3.8/site-packages/torchvision/transforms/functional.py


RUN source activate ldm && pip install clip taming-transformers-rom1504
# RUN source activate && conda activate ldm && python scripts/txt2img.py --prompt "a photograph of an astronaut riding a horse" --plms && rm -rf outputs


# RUN source activate && conda activate ldm && mkdir -p ghexample && wget --output-document ghexample/example.jpg https://github.com/CompVis/stable-diffusion/blob/main/assets/stable-samples/img3img/sketch-mountains-input.jpg && python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img ghexample/example.jpg --strength 0.8 && rm -rf outputs
```
After that the file structure should look as follows:
```
jianyus:/path/to/your/aigc-jobs$ ls
Dockerfile
```

3. Run docker build command:
```
jianyus@x99ews-10g-zs:docker build -t ruxiliang/aigc-example:1.0 .
```
4. Create a python file named time-eval.py so that you can get some initial result of the time consumption and memory consumption of the AIGC model:
```python
#! /root/miniconda3/bin/python -u
import subprocess
import time
import functools
import torch
import time
from pynvml import *
nvmlInit()


def exec_command(test_name:str, cmd:str):
    print(f'executing {test_name} for 100 times')
    times = []
    max_gpu_usage = -1
    for idx in range(10):
        start_time = time.time()
        p = subprocess.Popen(cmd,shell=True)
        while p.poll() is None:
            h1 = nvmlDeviceGetHandleByIndex(0)
            h2 = nvmlDeviceGetHandleByIndex(1)
            info1 = nvmlDeviceGetMemoryInfo(h1)
            info2 = nvmlDeviceGetMemoryInfo(h2)
            t = info1.total + info2.total
            a = info1.used + info2.used
            max_gpu_usage = max(max_gpu_usage, (a/t)*100)
        times.append(time.time() - start_time)
        print(f'iter {idx} costs {times[-1]} sec, the maximum gpu usage is {max_gpu_usage}')
    print(f'{test_name} costs {functools.reduce(lambda x, y: x + y, times) / len(times)}, and the max gpu usage is {max_gpu_usage}')


exec_command('5 samples', 'python scripts/txt2img.py --prompt "the beatles band striding along a zebra crossing situated on Moon" --plms ')
exec_command('10 samples', 'python scripts/txt2img.py --prompt "the beatles band striding along a zebra crossing situated on Moon" --plms --n_samples 10')
exec_command('1 sample', 'python scripts/txt2img.py --prompt "the beatles band striding along a zebra crossing situated on Moon" --plms --n_samples 1')
exec_command('1 sample', 'python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img /Pictures/sketch-mountains-input.jpg --strength 0.8 --n_samples 1')
exec_command('5 sample', 'python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img /Pictures/sketch-mountains-input.jpg --strength 0.8 --n_samples 5')
# exec_command('10 sample', 'python scripts/img2img.py --prompt "A fantasy landscape, trending on artstation" --init-img /Pictures/sketch-mountains-input.jpg --strength 0.8 --n_samples 10')
```
5. Clone the stable-diffusion code and download weights:
```
# exec following commands in $AIGC_DIR
git clone https://github.com/CompVis/stable-diffusion
cd stable-diffusion
wget --directory-prefix ckpt -c https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
mkdir -p models/ldm/stable-diffusion-v1/
mv ckpt/sd-v1-4.ckpt models/ldm/stable-diffusion-v1/model.ckpt
```
6. In $AIGC_DIR, create a new folder named `gpt-j` and create gptj.py in `$AIGC_DIR/gpt-j/`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer,GPTJForCausalLM, pipeline
import torch


# load fp 16 model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
# create pipeline
gen = pipeline("text-generation",model=model,tokenizer=tokenizer,device=0)


# run prediction
res = gen("K-ON!’s story revolves around four (and later five) Japanese high school girls who join their school’s Light Music Club.")
print(res)
```
7. To further benefit the profiling job, you can use the nsys-eval.py provided as follows:
```python3
#!/usr/bin/env python3
import argparse
import os
import subprocess


# parse cli commands
parser = argparse.ArgumentParser()
parser.add_argument('--gpu-name', required=True)
parser.add_argument('--txt2img-batches', required=True)
parser.add_argument('--txt2img-prompt', required=True)
parser.add_argument('--img2img-prompt', required=True)
parser.add_argument('--img2img-batches', required=True)
parser.add_argument('--img-path', required=True)
args = parser.parse_args()


# switch to /aigc-jobs
try:
    os.chdir('/aigc-jobs')
    print("Changed directory to /aigc-jobs")
except Exception as e:
    print(f"Error: {e}")
    exit(1)


# create subdirectory
subdirs = ['results-{}'.format(args.gpu_name),
           'results-{}/sd-{}'.format(args.gpu_name, args.gpu_name),
           'results-{}/gptj-{}'.format(args.gpu_name, args.gpu_name)]


for subdir in subdirs:
    if not os.path.exists(subdir):
        os.makedirs(subdir)
        print(f"Created directory {subdir}")


# run txt2img job
os.chdir('/aigc-jobs/stable-diffusion')
txt2img_batches = args.txt2img_batches.split(',')


for batch in txt2img_batches:
    cmd = f"nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --event-sample=system-wide -o /aigc-jobs/results-{args.gpu_name}/sd-{args.gpu_name}/txt2img-{args.gpu_name}-{batch} python scripts/txt2img.py --prompt \"{args.txt2img_prompt}\" --plms --n_samples {batch}"
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


# run img2img job
img2img_batches = args.img2img_batches.split(',')


for batch in img2img_batches:
    cmd = f"nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --event-sample=system-wide -o /aigc-jobs/results-{args.gpu_name}/sd-{args.gpu_name}/img2img-{args.gpu_name}-{batch} python scripts/img2img.py --prompt \"{args.img2img_prompt}\" --init-img {args.img_path} --strength 0.8 --n_samples 1"
    print(f"Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


# run gptj job
os.chdir('/aigc-jobs/gpt-j')
cmd = f"nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --event-sample=system-wide -o gptj-{args.gpu_name} python gptj.py"
print(f"Executing: {cmd}")
subprocess.run(cmd, shell=True, check=True)
```
After this, your folder should look as follows:
```
jianyus:~/aigc-jobs$ ls
Dockerfile  gpt-j  nsys_eval.py  stable-diffusion  time-eval.py
```
8. Then start run the docker image:
```
docker run -d --name aigc-jobs -v $AIGC_DIR:/aigc-jobs -it --gpus all ruxiliang/aigc-example:1.0
docker attach aigc-jobs
```
The reason why we detach the docker image is that the first time during its inference it may download some data from internet, so by detaching it we can save some time downloading those configurations.
9. Now insider the docker image:
```
root:/# source activate ldm
root:/# cd aigc-jobs
root:/aigc-jobs# python python nsys_eval.py --gpu-name rtx8000 --txt2img-batches 1,5,10 --txt2img-prompt "the beatles band striding along a zebra crossing situated on Moon" --img2img-prompt "A fantasy landscape, trending on artstation" --img-path  /aigc-jobs/Pictures/sketch-mountains-input.jpg  --img2img-batches 1,3
```