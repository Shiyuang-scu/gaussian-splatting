import torch 
import subprocess


scene = "drjohnson"
source_root = f"/home/yuang/Desktop/3d_gaussian_splat/dataset/source/db/{scene}"
output_root = f"/home/yuang/Desktop/3d_gaussian_splat/dataset/pre-trained_model/{scene}"
resolution_scales = [2, 4, 8]

train_script = "/home/yuang/Desktop/3d_gaussian_splat/gaussian-splatting/train.py"
render_script = "/home/yuang/Desktop/3d_gaussian_splat/gaussian-splatting/render.py"
eva_script = "/home/yuang/Desktop/3d_gaussian_splat/gaussian-splatting/metrics.py"

save_iter_list = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]
test_iter_list = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]
checkpoint_iter_list = [5_000, 10_000, 15_000, 20_000, 25_000, 30_000]

for res_scale in resolution_scales:
    source_dir = f"{source_root}_{res_scale}/"
    output_dir = f"{output_root}/{res_scale}/"
    
    '''Train 3DGS based on datasets with different resolutions.
    '''
    print(f"Training 3DGS with resolution scale {res_scale}...")
    render_command = [
        'python', render_script,
        '-s', source_dir,
        '-m', output_dir,
        '--data_device', 'cpu',
        '--save_iterations'] + [str(iteration) for iteration in save_iter_list] + \
        ['--test_iterations'] + [str(iteration) for iteration in test_iter_list] + \
        ['--checkpoint_iterations'] + [str(iteration) for iteration in checkpoint_iter_list] \
        
    subprocess.run(render_command)
    torch.cuda.empty_cache()


    ''' Render 2D images of 3DGS.
    '''
    print(f"Rendering 3DGS with resolution scale {res_scale}...")
    render_command = [
        'python', render_script,
        '-m', output_dir,
        ]
    subprocess.run(render_command)
    torch.cuda.empty_cache()


    '''Evaluate 3DGS: N.B. ground-truth images should come from the original dataset, instead of the downsampled dataset.
    '''
    print(f"Evaluating 3DGS with resolution scale {res_scale}...")
    eva_command = [
        'python', eva_script,
        '-m', output_dir,
        ]
    subprocess.run(eva_command)
    torch.cuda.empty_cache()
