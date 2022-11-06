# # install ros-noetic
# sudo apt-get update
# sudo apt-get upgrade -y
# sudo apt-get install lsb-release -y
# sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# sudo apt-get install curl wget git -y # if you haven't already installed curl
# sudo curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
# sudo apt-get update
# sudo apt-get install ros-noetic-desktop-full -y

# # # g++のバージョンアップデート
# sudo apt-get install software-properties-common -y
# add-apt-repository ppa:ubuntu-toolchain-r/test
# sudo apt-get update
# sudo apt-get upgrade -y
# sudo apt install python3-pip -y

# # cuda 11.1の中身追加？
# sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# sudo wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
# sudo apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda-11.1

# # omni3dの環境構築
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