<div align="center">

# A Preliminary Evaluation of the Non-determinism of VLAs for Robotic Tasks

Antonin Rousseau<sup>1</sup>,
Deyun Lyu<sup>2</sup>,
Paolo Arcaini<sup>2</sup>,
Fuyuki Ishikawa<sup>2</sup>,
Gilles Perrouin<sup>1</sup>
</br>
University of Namur, Namur, Belgium<sup>1</sup>, National Institute of Informatics, Tokyo, Japan<sup>2</sup>


</div>

## Data Availability

Testing scenes are generated and evaluated through the scripts found under '/experiments' with seed (4200, 4201, 5000, 6000).


# The base of this project was forked from [VLA_UQ](https://github.com/pablovalle/VLA_UQ)

## Prerequisites:
- CUDA version >=12.
- Cuda toolkit (nvcc)
- An NVIDIA GPU.
- Python >= 3.10
- Vulkan 
- Anaconda or environment virtualization tool

Clone this repo:
```
git clone https://github.com/pablovalle/VLA_UQ.git
```

## Installation for each VLA
Each VLA needs it's own dependencies and libraries, so we opted to generate a virtual environment for each of them. First, follow the following steps that are common for all the environments.

Create an anaconda environment:
```
conda create -n <env_name> python=3.10 (any version above 3.10 should be fine)
conda activate <env_name>
```

Install numpy<2.0 (otherwise errors in IK might occur in pinocchio):
```
pip install numpy==1.24.4
```

Install ManiSkill2 real-to-sim environments and their dependencies:
```
cd {this_repo}/ManiSkill2_real2sim
pip install -e .
```

Install this package:
```
cd {this_repo}
pip install -e .
```

```
sudo apt install ffmpeg
```

```
pip install tensorflow==2.15.0
pip install -r requirements_full_install.txt
pip install tensorflow[and-cuda]==2.15.1 # tensorflow gpu support
```

Install simulated annealing utils for system identification:
```
pip install git+https://github.com/nathanrooy/simulated-annealing
```

Install torch dependencies:
```
pip install torch==2.3.1 torchvision==0.18.1 timm==0.9.10 tokenizers==0.15.2 accelerate==0.32.1
pip install flash-attn==2.6.1 --no-build-isolation
```

Proceed with specific installation for each model in our evaluation:



### OpenVLA
Activate the conda environment:
```
conda activate <env_name>
```

Install the transformers (v.4.40.1) library in this repo:
```
cd {this_repo}/transformers-4.40.1
pip install -e .
```

Download the model:
```
cd {this_repo}/checkpoints
python download_model.py openvla/openvla-7b
```

Update the modeling_prismatic.py:
```
cd {this_repo}/checkpoints
cp modeling_prismatic openvla-7b
```



### pi0
Activate the conda environment:
```
conda activate <env_name>
```

Install the lerobot environment and its packages:
```
cd {this_repo}/lerobot
pip install -e .
```

Install the correct version of numpy, since it was changed:
```
pip install numpy==1.24.4
```

Install pytest to avoid possible errors:
```
pip install pytest
```

Install the transformers (v.4.48.1) library in this repo:
```
cd {this_repo}/transformers-4.48.1
pip install -e .
```

Download the model:
```
cd {this_repo}/checkpoints
python download_model.py HaomingSong/lerobot-pi0-bridge
python download_model.py HaomingSong/lerobot-pi0-fractal
```



### SpatialVLA
Activate the conda environment:
```
conda activate <env_name>
```

Install spatialVLA's requirements:
```
pip install -r spatialVLA_requirements
```

Install the transformers (v.4.48.1) library in this repo:
```
cd {this_repo}/transformers-4.48.1
pip install -e .
```
Download the model:
```
cd {this_repo}/checkpoints
python download_model.py IPEC-COMMUNITY/spatialvla-4b-mix-224-pt
```

Update the modeling_spatialvla.py:
```
cd {this_repo}/checkpoints
cp modeling_spatialvla spatialvla-4b
```

### Gr00t N1
Launch the models separately for SimplerENV following the instructions of the [original repo](https://github.com/NVIDIA/Isaac-GR00T/blob/n1d6/examples/SimplerEnv/README.md) (GR00T N1.6 checkpoint was used in the paper)

Activate the conda environment:
```
conda activate <env_name>
```

Follow earlier instructions for pi0

[Install client side Gr00t N1's requirements](https://github.com/NVIDIA/Isaac-GR00T/blob/n1d6/getting_started/policy.md#integrating-the-gr00t-16-client-into-your-deployment-pipeline)

## For Blackwell usage
At the end of the earlier instructions for each model:
```
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install -U https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.7.4%2Bcu128torch2.10-cp310-cp310-linux_x86_64.whl
```

## Replication Package

To reproduce experiment results, set the seed in the corresponding model script and launch:
```
cd experiments
./approach_random_gr00t_start.sh
```

```
cd experiments
./approach_random_spatialvla_start.sh
```

### Analyzing results

We provide notebooks used for analysis under '/notebooks'.


## Troubleshooting

1. If you encounter issues such as

```
RuntimeError: vk::Instance::enumeratePhysicalDevices: ErrorInitializationFailed
Some required Vulkan extension is not present. You may not use the renderer to render, however, CPU resources will be still available.
Segmentation fault (core dumped)
```

Follow [this link](https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/installation.html#vulkan) to troubleshoot the issue. (Even though the doc points to SAPIEN 3 and ManiSkill3, the troubleshooting section still applies to the current environments that use SAPIEN 2.2 and ManiSkill2).

2. You can ignore the following error if it is caused by tensorflow's internal code. Sometimes this error will occur when running the inference or debugging scripts.

```
TypeError: 'NoneType' object is not subscriptable
```

3. Please also refer to the original repo or [vulkan_setup](https://github.com/SpatialVLA/SpatialVLA/issues/3#issuecomment-2641739404) if you encounter any problems.

4. `tensorflow-2.15.0` conflicts with `tensorflow-2.15.1`?
The dlimp library has not been maintained for a long time, so the TensorFlow version might be out of date. A reliable solution is to comment out tensorflow==2.15.0 in the requirements file, install all other dependencies, and then install tensorflow==2.15.0 finally. Currently, using tensorflow==2.15.0 has not caused any problems.

## Citation

If you found our paper/code useful in your research, please consider citing:

```

```

## Acknowledgement

- [VLA_UQ](https://github.com/pablovalle/VLA_UQ)
- [SimplerEnv](https://github.com/DelinQu/SimplerEnv-OpenVLA)
- [ManiSkill2_real2sim](https://github.com/simpler-env/ManiSkill2_real2sim/tree/cd45dd27dc6bb26d048cb6570cdab4e3f935cc37)
- [VLATest](https://github.com/ma-labo/VLATest)

## Contact

