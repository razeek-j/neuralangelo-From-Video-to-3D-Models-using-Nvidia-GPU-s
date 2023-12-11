# Work in Progress

# Neuralangelo
This is the fork of official **Neuralangelo: High-Fidelity Neural Surface Reconstruction**.
To help installation on Windows Machine

### [Project page](https://research.nvidia.com/labs/dir/neuralangelo/) | [Paper](https://arxiv.org/abs/2306.03092/) | [Colab notebook](https://colab.research.google.com/drive/13u8DX9BNzQwiyPPCB7_4DbSxiQ5-_nGF)

<img src="assets/teaser.gif">

The code is built upon the Imaginaire library from the Deep Imagination Research Group at NVIDIA.  
For business inquiries, please submit the [NVIDIA research licensing form](https://www.nvidia.com/en-us/research/inquiries/).

--------------------------------------

## Installation
We offer to setup the environment:
1. We provide this through WSL (Windows Subsystem for Linux)
    - First install CUDA 11.8 Software on the Windows Machine (Nvidia Driver installation).
    - Then install WSL on the system by opening CMD and type `wsl --intsall` then restart and setup linux.
    - Then Close the CMD and open again and type `wsl` which it will go to WSL.
    - After that install Miniconda by pasting each command one by one or visit the website for future changes on how to install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
    ```bash
    mkdir -p ~/miniconda3
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
    rm -rf ~/miniconda3/miniconda.sh
    ~/miniconda3/bin/conda init bash
    ```
    - After installation Close the CMD and open it again with `wsl` command this it will be showing (base).
    - now lets install some dependenties by
    ```bash
    sudo apt update && sudo apt upgrade
    sudo apt-get install build-essential git g++
    ```
    - Then download or clone the project and open the location on CMS WSL environment using cd command.
    - The we create neuralangelo environment using commands

    ```bash
    conda env create --file neuralangelo.yaml
    conda activate neuralangelo
    ```
    - From (base) to (neuralangelo) after these commands that means you successfully activated the environment.
    - Then execute these commands
    ```bash
    export LIBRARY_PATH="/usr/lib/wsl/lib:$LIBRARY_PATH"
    pip install -r requirements.txt
    ```
    - But if you find any error at this stage mainly tinycuda issues it will be because of pytorch
    - So you need to install pytorch and verify it is working with cuda and it is connected with your nvidia GPU
    - To install and check follow this command
    ```bash
    pip install torch --no-cache-dir
    python3
    ```
    - We will check in the python(>>> it will show this symbol in terminal) by typing this code.
    ```bash
    import torch
    torch.cuda.get_device_name(0)
    ```
    - It should show your GPU name if not update your GPU drivers on windows install CUDA on Windows then redo the process
    - So after pip install -r requirments.txt without errors we will go forward to data preprocessing.
    - `docker.io/chenhsuanlin/neuralangelo:23.04-py3` is for running the main Neuralangelo pipeline.

    The corresponding Dockerfiles can be found in the `docker` directory.


--------------------------------------

## Data preparation
Please refer to [Data Preparation](DATA_PROCESSING.md) for step-by-step instructions.  
We assume known camera poses for each extracted frame from the video.
The code uses the same json format as [Instant NGP](https://github.com/NVlabs/instant-ngp).

--------------------------------------

## Run Neuralangelo!
```bash
EXPERIMENT=toy_example
GROUP=example_group
NAME=example_name
CONFIG=projects/neuralangelo/configs/custom/${EXPERIMENT}.yaml
GPUS=1  # use >1 for multi-GPU training!
torchrun --nproc_per_node=${GPUS} train.py \
    --logdir=logs/${GROUP}/${NAME} \
    --config=${CONFIG} \
    --show_pbar
```
Some useful notes:
- This codebase supports logging with [Weights & Biases](https://wandb.ai/site). You should have a W&B account for this.
    - Add `--wandb` to the command line argument to enable W&B logging.
    - Add `--wandb_name` to specify the W&B project name.
    - More detailed control can be found in the `init_wandb()` function in `imaginaire/trainers/base.py`.
- Configs can be overridden through the command line (e.g. `--optim.params.lr=1e-2`).
- Set `--checkpoint={CHECKPOINT_PATH}` to initialize with a certain checkpoint; set `--resume` to resume training.
- If appearance embeddings are enabled, make sure `data.num_images` is set to the number of training images.

--------------------------------------

## Isosurface extraction
Use the following command to run isosurface mesh extraction:
```bash
CHECKPOINT=logs/${GROUP}/${NAME}/xxx.pt
OUTPUT_MESH=xxx.ply
CONFIG=logs/${GROUP}/${NAME}/config.yaml
RESOLUTION=2048
BLOCK_RES=128
GPUS=1  # use >1 for multi-GPU mesh extraction
torchrun --nproc_per_node=${GPUS} projects/neuralangelo/scripts/extract_mesh.py \
    --config=${CONFIG} \
    --checkpoint=${CHECKPOINT} \
    --output_file=${OUTPUT_MESH} \
    --resolution=${RESOLUTION} \
    --block_res=${BLOCK_RES}
```
Some useful notes:
- Add `--textured` to extract meshes with textures.
- Add `--keep_lcc` to remove noises. May also remove thin structures.
- Lower `BLOCK_RES` to reduce GPU memory usage.
- Lower `RESOLUTION` to reduce mesh size.

--------------------------------------

## Frequently asked questions (FAQ)
1. **Q:** CUDA out of memory. How do I decrease the memory footprint?  
    **A:** Neuralangelo requires at least 24GB GPU memory with our default configuration. If you run out of memory, consider adjusting the following hyperparameters under `model.object.sdf.encoding.hashgrid` (with suggested values):

    | GPU VRAM      | Hyperparameter          |
    | :-----------: | :---------------------: |
    | 8GB           | `dict_size=20`, `dim=4` |
    | 12GB          | `dict_size=21`, `dim=4` |
    | 16GB          | `dict_size=21`, `dim=8` |

    Please note that the above hyperparameter adjustment may sacrifice the reconstruction quality.

   If Neuralangelo runs fine during training but CUDA out of memory during evaluation, consider adjusting the evaluation parameters under `data.val`, including setting smaller `image_size` (e.g., maximum resolution 200x200), and setting `batch_size=1`, `subset=1`.

2. **Q:** The reconstruction of my custom dataset is bad. What can I do?  
    **A:** It is worth looking into the following:
    - The camera poses recovered by COLMAP may be off. We have implemented tools (using [Blender](https://github.com/mli0603/BlenderNeuralangelo) or [Jupyter notebook](projects/neuralangelo/scripts/visualize_colmap.ipynb)) to inspect the COLMAP results.
    - The computed bounding regions may be off and/or too small/large. Please refer to [data preprocessing](DATA_PROCESSING.md) on how to adjust the bounding regions manually.
    - The video capture sequence may contain significant motion blur or out-of-focus frames. Higher shutter speed (reducing motion blur) and smaller aperture (increasing focus range) are very helpful.

--------------------------------------

## Citation
If you find our code useful for your research, please cite
```
@inproceedings{li2023neuralangelo,
  title={Neuralangelo: High-Fidelity Neural Surface Reconstruction},
  author={Li, Zhaoshuo and M\"uller, Thomas and Evans, Alex and Taylor, Russell H and Unberath, Mathias and Liu, Ming-Yu and Lin, Chen-Hsuan},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition ({CVPR})},
  year={2023}
}
```
