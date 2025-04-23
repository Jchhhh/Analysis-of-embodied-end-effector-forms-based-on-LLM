# Analysis-of-embodied-end-effector-forms-based-on-LLM
This study proposes an end-to-end system combining LLMs and generative AI to convert natural language into 3D-printable robotic end-effector models. Through semantic parsing, image generation, and 3D reconstruction, it improves design efficiency, accuracy, and printability, enabling intelligent automated design.

## Environment Configuration
### Robosuite Setup
1. Install robosuite with required dependencies  
pip install robosuite==1.4.0  
2. Install PyTorch (CUDA-enabled version recommended)  
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  
3. Clone and set up ComfyUI for generative workflows  
git clone https://github.com/comfyanonymous/ComfyUI.git  
cd ComfyUI && pip install -r requirements.txt  
