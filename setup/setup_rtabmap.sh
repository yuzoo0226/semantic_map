sudo cd ~
mkdir Library
cd Library

git clone https://github.com/introlab/rtabmap.git rtabmap
cd rtabmap/build
cmake .. 
make
sudo make install

sudo apt-get install ros-noetic-move-base -y

# cd ~/catkin_ws
# git clone https://github.com/introlab/rtabmap_ros.git src/rtabmap_ros
# catkin_make -j1