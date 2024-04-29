import torch 
import subprocess
from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths, target_dir, gt_dir, trainOrTest="test"):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                renders, gts, image_names = readImages(target_dir, gt_dir)

                ssims = []
                psnrs = []
                lpipss = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))
                    psnrs.append(psnr(renders[idx], gts[idx]))
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            with open(f"{scene_dir}/{trainOrTest}_results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(f"{scene_dir}/{trainOrTest}_per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except:
            print("Unable to compute metrics for model", scene_dir)


scene = "drjohnson"
source_root = f"/home/yuang/Desktop/3d_gaussian_splat/dataset/source/db/{scene}"
output_root = f"/home/yuang/Desktop/3d_gaussian_splat/dataset/pre-trained_model/{scene}"
gt_images_dir = f"/home/yuang/Desktop/3d_gaussian_splat/dataset/pre-trained_model/{scene}"

resolution_scales = [1, 2, 4, 8]

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
        'python', train_script,
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
    if res_scale == 1:
        render_command = [
            'python', render_script,
            '-m', output_dir,
            ]
    else:
        render_command = [
            'python', render_script,
            '-m', output_dir,
            '--skip_gt',
            ]
    subprocess.run(render_command)
    torch.cuda.empty_cache()


    '''Evaluate 3DGS: N.B. ground-truth images should come from the original dataset, instead of the downsampled dataset.
    '''
    print(f"Evaluating 3DGS with resolution scale {res_scale}...")
    # eva_command = [
    #     'python', eva_script,
    #     '-m', output_dir,
    #     ]
    # subprocess.run(eva_command)
    
    train_target_dir = f"{output_dir}/train/ours_30000/renders" 
    test_target_dir = f"{output_dir}/test/ours_30000/renders" 
    
    train_gt_dir = f"{output_root}/1/train/ours_30000/gt"
    test_gt_dir = f"{output_root}/1/test/ours_30000/gt"
    
    evaluate([output_dir], train_target_dir, train_gt_dir, trainOrTest="train")
    evaluate([output_dir], test_target_dir, test_gt_dir, trainOrTest="test")

    torch.cuda.empty_cache()
