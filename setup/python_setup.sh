# omni3dの環境構築
# conda create -n cubercnn python=3.9
# conda activate cubercnn

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install --user 'git+https://github.com/facebookresearch/fvcore'

pip install "git+https://github.com/facebookresearch/pytorch3d.git"

git clone https://github.com/facebookresearch/detectron2.git
pip install --user -e detectron2
# rm -rf detectron2
pip install scipy pandas opencv-python
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

pip install rospkg