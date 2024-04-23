<p align="center">
    <img src="figs/Logo2.png" width="30%", style="vertical-align: middle; margin-right: 0px; position: relative; top: 50px;"> <br>

</p>

<div align="center">

## FlashFace: Human Image Personalization with High-fidelity Identity Preservation
</div> 
<!-- ## <center>🔥[**Online Demo**](<>)🔥    🔥[**Project Page**](https://jshilong.github.io/flashface-page/)🔥 </center> -->

##   [**Project Page**](https://jshilong.github.io/flashface-page/) 


## Feature

1. Zero-shot human image customization within seconds with one or several reference faces.
2. Strong identity preservation ability, even for non-celebrities
3. Strong language prompts following ability, even changing the age or gender of the person.
4. Flexible strength adjustment for identity image control and language prompt control.

## Updates

<p align="center">
<img src="https://github.com/jshilong/flashface-page/blob/main/static/githubfigs/flashface.gif" width="100%"> <br>


</p>

- \[April 18\] Inference code of **FlashFace-SD1.5** are released.

This version represents a clear advancement over our paper, as it has been trained for an extended duration. We also involve new guidance techniques in the inference process. Notably, this enhanced checkpoint demonstrates remarkable progress in terms of lighting and shadow effects. It show strong identity preservation ability even for non-celebrities. Please refer to the following images for more details. For the prompt and inference parameters, please refer to [**notebook demo**](flashface/all_finetune/inference.ipynb). If you are not satisfied with the generated images, you can submit your case in the issue so that we can help you improve it. Stay tuned for more versions!

我们开源了 FlashFace-SD1.5 的 inference 代码以及全微调的模型权重. 这个权重相比我们论文中权重训练了更长的时间，如下面图片，展现出了更真实的光影效果与人物细节,即使针对非名人它也展现出了极强的保 ID 能力. 下面图片使用的 prompt 以及 inference 参数细节 可以参照 [**notebook demo**](flashface/all_finetune/inference.ipynb). 如果您对生成图片不满意，可以在 issue 中提交您的 case,以便我们帮助您提升它. 其他版本权重敬请期待！

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
