#!/bin/bash
# vision
# image classification
wget -O resnet18.pt https://www.dropbox.com/s/w8kz4lm3wkrmif9/resnet18.pt?dl=1
wget -O googlenet.pt https://www.dropbox.com/s/ur47zj3vtn2zizb/googlenet.pt?dl=1
wget -O alexnet.pt https://www.dropbox.com/s/qsrexk82d2np2sw/alexnet.pt?dl=1
wget -O mnasnet0_5.pt https://www.dropbox.com/s/1fp31v8l0pa22nn/mnasnet0_5.pt?dl=1
wget -O mobilenet_v2.pt https://www.dropbox.com/s/ee2l1rl6hodyank/mobilenet_v2.pt?dl=1
wget -O inceptionv3.pt https://www.dropbox.com/s/zmynwuk72cregtv/inceptionv3.pt?dl=1
wget -O densenet121.pt https://www.dropbox.com/s/4a7jrl9j2i3axxi/densenet121.pt?dl=1
wget -O resnext50_32x4d.pt https://www.dropbox.com/s/ur39h6iamlg7j0z/resnext50_32x4d.pt?dl=1
wget -O shufflenet_v2_x0_5.pt https://www.dropbox.com/s/x3u1scqbxy77vdm/shufflenet_v2_x0_5.pt?dl=1
wget -O squeezenet1_0.pt https://www.dropbox.com/s/fr97gdv4vbss6hf/squeezenet1_0.pt?dl=1
wget -O vgg11.pt https://www.dropbox.com/s/w6g81etb4nhatkd/vgg11.pt?dl=1
#image segmentation
wget -O deeplabv3resnet101.pt https://www.dropbox.com/s/erfgswv5c3cw30k/deeplabv3resnet101.pt?dl=1
wget -O fcnresnet101.pt https://www.dropbox.com/s/5job5jewmohj83v/fcnresnet101.pt?dl=1
# nlp
# sentiment analysis
wget -O electra_imdb.pt https://www.dropbox.com/s/dlggn5nol3s3dcf/electra_imdb.pt?dl=1
wget -O distilbert_imdb.pt https://www.dropbox.com/s/eci8ql4a7wnfyfp/distilbert_imdb.pt?dl=1
# move all to assets
mv *.pt android/app/src/main/assets/