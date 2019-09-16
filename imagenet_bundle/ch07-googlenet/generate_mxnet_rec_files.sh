~/mxnet/bin/im2rec /raid/datasets/imagenet/lists/train.lst "" \
    /raid/datasets/imagenet/rec/train.rec \
    resize = 256 encoding = '.jpg' \
    quality = 100

~/mxnet/bin/im2rec /raid/datasets/imagenet/lists/val.lst "" \
    /raid/datasets/imagenet/rec/val.rec \
    resize = 256 encoding = '.jpg' \
    quality = 100