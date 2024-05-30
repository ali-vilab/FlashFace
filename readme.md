<p align="center">
    <img src="figs/Logo2.png" width="30%", style="vertical-align: middle; margin-right: 0px; position: relative; top: 50px;"> <br>

</p>

<div align="center">

## FlashFace: Human Image Personalization with High-fidelity Identity Preservation
</div> 

###  [🔥ComfyUI-FlashFace🔥](https://github.com/cold-hand/ComfyUI-FlashFace)（Thanks to [Cold-Hand](https://github.com/cold-hand)'s excellent work, they have discovered that ``adding more reference faces(instead default 4 faces in this repo) can further enhance the FlashFace``. This implementation is still in the process of rapid iteration and feature addition.） 
 
### 🔥 [**Project Page**](https://jshilong.github.io/flashface-page/)  🔥

###  [🔥Unofficial Online Demo🔥](https://replicate.com/zsxkib/flash-face)（please use the following hyper-parameters instead of the default hyper-parameters of this demo to obtain stable ID Fidelity） 



#### Recommended hyper-parameters to obtain stable ID Fidelity
```
# Please include the age word in the prompt, e.g. young woman/man
# Otherwise, FlashFace tends to produce middle-aged faces, which tend to be fatter.
# 请在 prompt 包含年龄词如 young，否则 FlashFace 可能生成中年照片，脸部一般偏胖


positive prompt: A handsome young man / A beautiful young woman .......
face position: [0.3, 0.2, 0.6, 0.5] # avoid generating faces that are too large or too small
Reference Feature Strength: 1.2
Reference Guidance Strength: 3.2
Step Index to Launch Ref Guidance: 750

# When artifacts are found on the face, reduce these three values appropriately.
# 当脸部出现贴图感,请适当调低这三个参数
```
<p align="center">
<img src="https://github.com/jshilong/flashface-page/blob/main/static/githubfigs/output1-min.png" width="80%"> <br>

</p>

#### Recommended hyper-parameters to change the age
```
# Please include age words in the prompt, e.g. baby girl/boy,  An very old woman/man


positive prompt: A baby girl / An very old woman ......
face position: [0.3, 0.2, 0.6, 0.5] # avoid generating faces that are too large or too small
default_text_control_scale = 8.5

Reference Feature Strength: 0.9
Reference Guidance Strength: 2.5
Step Index to Launch Ref Guidance: 750

# When ID Fidelity is not enough for the changed-aged person, please turn these values up
# 如果感觉人脸相似度不够，请调高这三个参数

```
<p align="center">
<img src="https://github.com/jshilong/flashface-page/blob/main/static/githubfigs/output2-min.png" width="80%"> <br>
<img src="https://github.com/jshilong/flashface-page/blob/main/static/githubfigs/output3-min.png" width="80%"> <br>

</p>


###   为了获得更好的结果，特别是对于亚洲人脸， 请在使用前您先行阅读 [中文教程](./docs/zh_cn.md) 获取一些经验，否则可能会导致生成结果较差。 

###  For optimal results, especially for Asian faces, it's strongly advised to read through this [tutorial](./docs/en.md) before diving in. Skipping this step may lead to undesirable or unanticipated outcomes.  


## Updates



- \[May 30\] [🔥ComfyUI-FlashFace🔥](https://github.com/cold-hand/ComfyUI-FlashFace). Thanks to [Cold-Hand](https://github.com/cold-hand)'s excellent work, they find that ``adding more reference faces(instead default 4 faces in this repo) can further enhance the FlashFace``. This implementation is still in the process of rapid iteration and feature addition.

- \[May 7\] 🔥🔥🔥 An [Online Demo](https://replicate.com/zsxkib/flash-face) for **FlashFace-SD1.5** has been graciously provided by [Sakib Ahamed](https://github.com/zsxkib). To achieve optimal results, it's recommended to thoroughly go through this [tutorial](./docs/en.md) / [中文教程](./docs/zh_cn.md) before jumping in.




- \[April 18\] Inference code of **FlashFace-SD1.5** has been released.

This version represents a clear advancement over our paper, as it has been trained for an extended duration. We also involve new guidance techniques in the inference process. Notably, this enhanced checkpoint demonstrates remarkable progress in terms of lighting and shadow effects. It show strong identity preservation ability even for non-celebrities. Please refer to the following images for more details. For the prompt and inference parameters, please refer to [**notebook demo**](flashface/all_finetune/inference.ipynb). If you are not satisfied with the generated images, you can submit your case in the issue so that we can help you improve it. Stay tuned for more versions!

我们开源了 FlashFace-SD1.5 的 inference 代码以及全微调的模型权重. 这个权重相比我们论文中权重训练了更长的时间，如下面图片，展现出了更真实的光影效果与人物细节,即使针对非名人它也展现出了极强的保 ID 能力. 下面图片使用的 prompt 以及 inference 参数细节 可以参照 [**notebook demo**](flashface/all_finetune/inference.ipynb). 如果您对生成图片不满意，可以在 issue 中提交您的 case,以便我们帮助您提升它. 其他版本权重敬请期待！


## Feature

<p align="center">
<img src="https://github.com/jshilong/flashface-page/blob/main/static/githubfigs/flashface.gif" width="100%"> <br>

</p>

1. Zero-shot human image customization within seconds with one or several reference faces.
2. Strong identity preservation ability, even for non-celebrities
3. Strong language prompts following ability, even changing the age or gender of the person.
4. Flexible strength adjustment for identity image control and language prompt control.


<p align="center">
<img src="https://jshilong.github.io/flashface-page/static/githubfigs/genrate_afanda.png" width="120%"> <br>
<img src="https://jshilong.github.io/flashface-page/static/githubfigs/genrate_baixue.png" width="120%"> <br>
<img src="https://jshilong.github.io/flashface-page/static/githubfigs/genrate_alen.png" width="120%"> <br>
<img src="https://jshilong.github.io/flashface-page/static/githubfigs/genrate_man.png" width="120%"> <br>
<img src="https://jshilong.github.io/flashface-page/static/githubfigs/genrate_woman.png" width="120%"> <br>
<img src="https://jshilong.github.io/flashface-page/static/githubfigs/details.png" width="120%"> <br>

</p>

## Contents

- [Install](#Install)
- [FlashFace Weights](#Weights)
- [Demo](#Demo)
- [Acknowledge](#Acknowledge)

## Install

1. Clone the `FlashFace`

```python
git clone https://github.com/ali-vilab/FlashFace.git
cd FlashFace
```

2. Create the env

```shell
conda create -n flashface python=3.10 -y
conda activate flashface
pip install -r requirements.txt
```

## Weights

You can download the weights from the ModelScope or Huggingface.

### ModelScope

link: [ModelScope-FlashFace](https://www.modelscope.cn/models/iic/FlashFace/summary)

### Huggingface

link: [Huggingface-FlashFace](https://huggingface.co/shilongz/FlashFace-SD1.5)

You should download all the related weights and put them in the `cache` folder (e.g. `FlashFace/cache/flashface.ckpt`).

## Demo

You can do the inference with the [**notebook demo**](flashface/all_finetune/inference.ipynb) or Gradio.

```python
python flashface/all_finetune/demo_gradio.py
```

## Acknowledgement

Inspired from many excellent demos and repos related to human image customization, including [IP-Adapter](https://github.com/tencent-ailab/IP-Adapter), [multimodalart/Ip-Adapter-FaceID](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID), [FastComposer](https://github.com/mit-han-lab/fastcomposer), [T2I-Adapter](https://github.com/TencentARC/T2I-Adapter), [PhotoMaker](https://github.dev/TencentARC/PhotoMaker) and [InstantID](https://github.com/InstantID/InstantID). Thanks for their great work!

## Disclaimer

This project aims to have a positive impact on the domain of AI-driven image generation.
Users are granted the freedom to create images using this tool, but they are expected to comply with local laws and use it responsibly.
The developers do not assume any responsibility for potential misuse by users.

If you find FlashFace useful for your your research and applications, please cite using this BibTeX:

```bibtex
@misc{zhang2024flashface,
      title={FlashFace: Human Image Personalization with High-fidelity Identity Preservation},
      author={Shilong Zhang and Lianghua Huang and Xi Chen and Yifei Zhang and Zhi-Fan Wu and Yutong Feng and Wei Wang and Yujun Shen and Yu Liu and Ping Luo},
      year={2024},
      eprint={2403.17008},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
