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
