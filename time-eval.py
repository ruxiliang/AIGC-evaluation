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