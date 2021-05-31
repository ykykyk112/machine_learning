import os
import wget


def bar_custom(current, total, width=80):
    progress = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    return progress


def download_imagenet(root='D:\ImageNet'):
    """
    download_imagenet validation set
    :param img_dir: root for download imagenet
    :return:
    """

    # make url
    val_url = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_train.tar'
    devkit_url = 'http://www.image-net.org/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar'

    print("Download...")
    os.makedirs(root, exist_ok=True)
    wget.download(url=val_url, out=root, bar=bar_custom)
    print('')
    wget.download(url=devkit_url, out=root, bar=bar_custom)
    print('')
    print('done!')

if __name__ == '__main__' :
    download_imagenet()