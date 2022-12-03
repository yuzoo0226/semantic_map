### cuda 11.1
# singularity build --sandbox hsr_ros docker://nvidia/cuda:11.1.1-runtime-ubuntu20.04

apt-get update
apt-get upgrade -y 

apt-get install curl wget git -y # if you haven't already installed curl

# cuda 11.1の中身追加？
# nvccが必要ぽい
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
dpkg -i cuda-repo-ubuntu2004-11-1-local_11.1.1-455.32.00-1_amd64.deb
apt-key add /var/cuda-repo-ubuntu2004-11-1-local/7fa2af80.pub
apt-get update
apt-get -y install cuda-11.1

# install ros-noetic
sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
apt-get update
apt-get install ros-noetic-desktop-full -y

# g++のバージョンアップデート
apt-get install software-properties-common -y
add-apt-repository ppa:ubuntu-toolchain-r/test
apt-get update
apt-get upgrade -y
apt install python3-pip -y


# omni3dの環境構築​
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
export FORCE_CUDA=1
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
git clone https://github.com/facebookresearch/detectron2.git
python3 -m pip install -e detectron2
pip install scipy pandas opencv-python
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'

# rtabmapの環境構築
git clone https://github.com/introlab/rtabmap.git rtabmap
cd rtabmap/build
cmake .. 
make
make install
apt-get install ros-noetic-move-base -y

cd ~/hma_ws
git clone https://github.com/introlab/rtabmap_ros.git src/rtabmap_ros
catkin_make


# # test for omni3d
# cd ~
# git clone https://github.com/facebookresearch/omni3d.git
# cd omni3d
# # Download example COCO images
# sh demo/download_demo_COCO_images.sh

# # Run an example demo
# python3 demo/demo.py \
# --config-file cubercnn://omni3d/cubercnn_DLA34_FPN.yaml \
# --input-folder "datasets/coco_examples" \
# --threshold 0.25 --display \
# MODEL.WEIGHTS cubercnn://omni3d/cubercnn_DLA34_FPN.pth \
# OUTPUT_DIR output/demo 