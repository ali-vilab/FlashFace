# ComfyUI-FlashFace
ComfyUI implementation of FlashFace: Human Image Personalization with High-fidelity Identity Preservation </center>
> Officially implemented here: https://github.com/ali-vilab/FlashFace

## Installation </center>
1. Navigate to the `custom_nodes` directory in your ComfyUI installation.
2. Install the custom node by running `git clone https://github.com/cold-hand/ComfyUI-FlashFace.git` in the `custom_nodes` directory.
3. If you are using a virtual environment, activate it.
4. cd into the `ComfyUI-FlashFace` directory and run `pip install -r requirements-comfy.txt` to install the required packages. Do not run `pip install -r requirements.txt` as this will install the requirements for the original FlashFace repository, and possible break your comfy installation by installing torch again
5. Restart ComfyUI and you should see the FlashFace node in the node list.
6. Install the ImpactPack from the ComfyUI Manager.
7. Install Facerestore CF (Code Former) from the ComfyUI Manager. 
8. Load the provided example-workflow.json file at the root of this repo to see how the nodes are used.

## Change Log
- [5/20/2024]: Initial release