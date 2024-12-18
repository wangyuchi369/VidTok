
# VidTwin: Video VAE with Decoupled Structure and Dynamics

**yuchi comments**: *把vidtwin的代码迁移到了该框架中，所以data上没变，就用的vidtok里更general的版本；主要是models添加了vidtwin的类，以及在modules里添加了所需的模块；*

*README中主要是用法，因此基本没什么改变，在scripts里添加了Cross-reenactment的脚本; diffusion部分感觉没啥必要放上去了，主要代码结构也比较复杂*

![vidtwin](assets/vidtwin.png)
We propose a novel and compact video autoencoder, VidTwin, that decouples video into two distinct latent spaces: Structure latent vectors, which capture overall content and global movement, and Dynamics latent vectors, which represent fine-grained details and rapid movements. 

Extensive experiments show that VidTwin achieves a high compression rate of 0.20% with high reconstruction quality (PSNR of 28.14 on the MCL-JCV dataset), and performs efficiently and effectively in downstream generative tasks. Moreover, our model demonstrates explainability and scalability, paving the way for future research in video latent representation and generation.
![vidtwin](assets/vidtwin_disen.png)

## Setup

**yuchi comments**: *same as VidTok. I have added some packages in enviroments.yaml*

## Checkpoints

**yuchi comments**: *put the ckpts in the same folder in Huggingface?*


## Training

### Data Preparation

**yuchi comments** *follows the way VidTok uses*

### Fine-tune on Custom Data

**yuchi comments** *same; apart from 'fix_encoder' setting*



## Inference

**yuchi comments** *almost same, apart from resolution, num-frames, etc*

### Easy Usage
We provide the following example for a quick usage of our models. 
Just provide the path to the configuration file `cfg_path` and checkpoint file `ckpt_path`.
```python
import torch
from scripts.inference_evaluate import load_model_from_config

cfg_path = "configs/vidtwin/vidtwin_structure_7_7_8_dynamics_7_8.yaml"
ckpt_path = "checkpoints/vidtwin_structure_7_7_8_dynamics_7_8.ckpt"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    
# load pre-trained model
model = load_model_from_config(cfg_path, ckpt_path)
model.to(device).eval()
# random input
num_frames = 16
x_input = (torch.rand(1, 3, num_frames, 224, 224) * 2 - 1).to(device)  # [B, C, T, H, W], range -1~1
# model forward
_, x_recon, _ = model(x_input)
assert x_input.shape == x_recon.shape
```

### Reconstruct an Input Video
```
python scripts/inference_reconstruct.py --config CONFIG --ckpt CKPT --input_video_path VIDEO_PATH --num_frames_per_batch NUM_FRAMES_PER_BATCH --input_height 224 --input_width 224 --sample_fps 30 --output_video_dir OUTPUT_DIR
```
- Specify `VIDEO_PATH` to the path of your test video. We provide an example video in `assets/example.mp4`. 
- Set `NUM_FRAMES_PER_BATCH` to `16.
- The reconstructed video is saved in `OUTPUT_DIR`.

### Performance Evaluation
We also provide a manuscript `scripts/inference_evaluate.py` to evaluate the video reconstruction performance in PSNR, SSIM and LPIPS.

1. Put all of your test videos under `DATA_DIR`.
2. Run the following command, and all `.mp4` videos under `DATA_DIR` will be tested:
```
python scripts/inference_evaluate.py --config CONFIG --ckpt CKPT --data_dir DATA_DIR --num_frames_per_batch NUM_FRAMES_PER_BATCH --input_height 256 --input_width 256 --sample_fps 30
```
(Optional) If you only want to test certain videos under `DATA_DIR`, you need to prepare a `.csv` meta file 
to indicate the video files to be tested (refer to [Data Preparation](#data-preparation)). And add `--meta_path META_PATH` to the above command to specify the path to the `.csv` meta file.

**yuchi comments** *add Cross-reenactment scripts*




### Cross-reenactment of VidTwin Model

For VidTwin model, we conduct a cross-reenactment experiment in which we combine the *Structure Latent* from one video, $A$, with the *Dynamics Latent* from another video, $B$, to observe the generated output from the decoder, i.e., generating $\mathcal{D}(u^A_{\boldsymbol{S}}, u^B_{\boldsymbol{D}})$.

To facilitate this experiment, we provide the script `scripts/inference_vidtwin_cross_reconstruct.py`. This script follows a similar usage method to `scripts/inference_reconstruct.py,` with the addition of two new arguments: `--input_video_path_structure` and `--input_video_path_dynamics`, which allow you to specify the videos for structure and dynamics information, respectively.


## BibTeX

```bibtex
@article{wang2024vidtwin,
  title={VidTwin: Video VAE with Decoupled Structure and Dynamics},
  author={},
  year={2024},
  journal={arXiv preprint arXiv:2412.xxxxx},
}
```
