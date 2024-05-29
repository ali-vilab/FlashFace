@echo off

pip install -r requirements-comfy.txt

cd ..\..\models
mkdir flashface
cd flashface
bitsadmin /transfer flashface_download https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/flashface.ckpt?download=true "%cd%\flashface.ckpt"

cd ..\vae
bitsadmin /transfer vae_download https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/sd-v1-vae.pth?download=true "%cd%\sd-v1-vae.pth"

cd ..\clip
bitsadmin /transfer clip_download1 https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/openai-clip-vit-large-14.pth?download=true "%cd%\openai-clip-vit-large-14.pth"
bitsadmin /transfer clip_download2 https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/bpe_simple_vocab_16e6.txt.gz?download=true "%cd%\bpe_simple_vocab_16e6.txt.gz"
cd ..
mkdir facedetection
cd facedetection
bitsadmin /transfer facedetection_download https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/retinaface_resnet50.pth?download=true "%cd%\retinaface_resnet50.pth"

pause