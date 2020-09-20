import torch.utils.data as data
import argparse
from img_read import *
from model import HRNet
import torch.nn.functional as F
from matplotlib.pylab import plt
import torch
import scipy.misc
from matplotlib import image
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--input_root", type=str, default="/home/z/下载/work1/datasets/EMOdImages1019/",
                    help="root path of images")

# parser.add_argument("--input_root", type=str, default="/home/z/下载/work1/datasets/CAT/",
#                     help="root path of images")

args = parser.parse_args()
data_test = Imageio_test(root0=args.input_root)
eval_loader = torch.utils.data.DataLoader(data_test, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

model = HRNet().eval().cuda()
model.load_state_dict(torch.load('./rms/my 6.5 mul model10.pth'))

with torch.no_grad():
    for i_batch, sample_batch in enumerate(eval_loader):
        input = sample_batch[0].cuda()
        name = sample_batch[1]
        print(name)
        a = name[0].split(".")
        # print(a[0])
        out = model(input)
        out = F.interpolate(out, size=(768, 1024), mode="bicubic", align_corners=True)
        out = out.cpu().view(768, 1024)
        out = out.detach().numpy()
        scipy.misc.toimage(out).save(
            '/home/z/下载/metrics_python/metrics_python/EMOd/' + str(a[0]) + ".jpg")
        # image.imsave('/home/z/下载/metrics_python/metrics_python/color/对比实验/cat attention CAT2000/'
        #              + str(a[0]) + ".jpg", out)

