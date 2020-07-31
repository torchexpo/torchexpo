#!/bin/bash
# vision
# image classification
wget -O resnet18.pt https://www.dropbox.com/s/49ydm1ibzvdudh0/resnet18.pt?dl=1
mv resnet18.pt android/app/src/main/assets/
wget -O googlenet.pt https://www.dropbox.com/s/hm3w0kh7eoha29f/googlenet.pt?dl=1
mv googlenet.pt android/app/src/main/assets/