pip install -r requirements-comfy.txt
cd ../../models
mkdir flashface
cd flashface
wget -O flashface.ckpt https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/flashface.ckpt?download=true
cd ../vae
wget -O sd-v1-vae.pth https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/sd-v1-vae.pth?download=true
cd ../clip
wget -O openai-clip-vit-large-14.pth https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/openai-clip-vit-large-14.pth?download=true
wget -O bpe_simple_vocab_16e6.txt.gz https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/bpe_simple_vocab_16e6.txt.gz?download=true
cd ..
mkdir facedetection
cd facedetection
wget -O retinaface_resnet50.pth https://huggingface.co/shilongz/FlashFace-SD1.5/resolve/main/retinaface_resnet50.pth?download=true