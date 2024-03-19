import subprocess
import os
import sys
from pathlib import Path
import time
import torch
'''
Training
python train.py 
-s {source_dir} 
-m {output_dir} 
--data_device {device} # "cpu" or "cuda"
--iterations {iteration_num} # 30_000 by default
--eval # Train with train/test split
--save_iterations {save_iter_list} # a list of iterations to save, [7_000, 30_000] by default
--test_iterations {test_iter_list} # a list of iterations to calculate the testing loss, [7_000, 30_000] by default
--checkpoint_iterations {checkpoint_iter_list} # a list of iterations to save the checkpoint
--start_checkpoint {checkpoint_path} # Path to a saved checkpoint to continue training from.
'''

'''
Rendering
python render.py 
-m {model_dir} 
--skip_train
# the following arguments will be read automatically from the model path
-s {source_dir} 
--data_device {device} 
'''

'''
Evaluating
python metrics.py 
-m {model_dir} 
'''





# Training
SUBSET = True # train with subset of the dataset 
# SUBSET = False # train with the whole dataset
if SUBSET:
    n = 1250
    source_dir = f"/home/yuang/Desktop/3d_gaussian_splat/dataset/source/eyefultower/apartment_{n}/"
    output_dir = f"/home/yuang/Desktop/3d_gaussian_splat/dataset/pre-trained_model/apartment/{n}/"
else:
    n = 4000
    source_dir = "/home/yuang/Desktop/3d_gaussian_splat/dataset/source/eyefultower/apartment/"
    output_dir = "/home/yuang/Desktop/3d_gaussian_splat/dataset/pre-trained_model/apartment/all/"

dataset_size = int(n*0.8)

prog_train_interval = 100
# iteration_num = n * 150
iteration_num = prog_train_interval

Path(output_dir).mkdir(parents=True, exist_ok=True) # if output_dir does not exist, create the directory
device = "cpu"
# device = "cuda"

save_iter_list = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]
# test_iter_list = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]
test_iter_list = [5000*i for i in range(1,dataset_size*150//5000+1)]
checkpoint_iter_list = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]
# checkpoint_path = os.path.join(output_dir, "chkpnt15000.pth")
train_script = "/home/yuang/Desktop/gaussian-splatting/train.py"

# If you have a Python program that uses argparse nargs to specify multi-value options 
# then you need to make sure that each value is its own parameter.
# https://gist.github.com/CMCDragonkai/0118e112e3d0de377f71bf92efc7ace6
command = [
    'python', train_script,
    '-s', source_dir,
    '-m', output_dir,
    '--data_device', device,
    '--iterations', str(iteration_num),
    '--eval',
    '--prog_train_interval', str(prog_train_interval),
    '--dataset_size', str(dataset_size),
    '--save_iterations'] + [str(iteration) for iteration in save_iter_list] + \
    ['--test_iterations'] + [str(iteration) for iteration in test_iter_list] + \
    ['--checkpoint_iterations'] + [str(iteration) for iteration in checkpoint_iter_list] \
    # + ['--start_checkpoint', checkpoint_path]

subprocess.run(command)
torch.cuda.empty_cache()


# Rendering and Evaluating
render_script = "/home/yuang/Desktop/gaussian-splatting/render.py"
eva_script = "/home/yuang/Desktop/gaussian-splatting/metrics.py"


command = [
    'python', render_script,
    '-m', output_dir,
    '--skip_train',
    ]
subprocess.run(command)
torch.cuda.empty_cache()


command = [
    'python', eva_script,
    '-m', output_dir,
    '-d', 'cpu',
    ]
subprocess.run(command)
torch.cuda.empty_cache()