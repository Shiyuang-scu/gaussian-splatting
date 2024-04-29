import torch 
import subprocess


scene = "drjohnson"
output_root = f"/home/yuang/Desktop/3D_gaussian_splat/dataset/pre-trained_model/{scene}"
resolution_scales = [1, 2, 4, 8]

render_script = "/home/yuang/Desktop/3D_gaussian_splat/render.py"
eva_script = "/home/yuang/Desktop/3D_gaussian_splat/metrics.py"
# Train 3DGS based on datasets with different resolutions.



for res_scale in resolution_scales:

    output_dir = f"{output_root}/{res_scale}/"

    # Render 2D images of 3DGS.
    print(f"Rendering 3DGS with resolution scale {res_scale}...")
    render_command = [
        'python', render_script,
        '-m', output_dir,
        ]
    subprocess.run(render_command)
    torch.cuda.empty_cache()

    # Evaluate 3DGS: N.B. ground-truth images should come from the original dataset, instead of the downsampled dataset.
    print(f"Evaluating 3DGS with resolution scale {res_scale}...")
    eva_command = [
        'python', eva_script,
        '-m', output_dir,
        ]
    subprocess.run(eva_command)
    torch.cuda.empty_cache()
