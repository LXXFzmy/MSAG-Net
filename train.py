from model import *
import argparse
import torch
from torch import optim
from model import HRNet
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from img_read import *
from data import *
from time import time
np.set_printoptions(threshold=np.inf)

parser = argparse.ArgumentParser()

parser.add_argument("--input_root", type=str, default="/home/z/下载/work1/SALICON/images/train/train/",
                    help="root path of images")
parser.add_argument("--gt_root", type=str, default="/home/z/下载/work1/SALICON/images/fixations_train2014/fixations_train2014/",
                    help="root path of gt")
parser.add_argument("--num_epoch", type=int, default=10,
                    help="number of epoch")
parser.add_argument("--lr", type=float, default=1e-5,
                    help="learning rate")
parser.add_argument("--save-freq", type=int, default=1,
                    help="frequency to save model")
args = parser.parse_args()

data = Imageio(root0=args.input_root, root1=args.gt_root)
train_loader = torch.utils.data.DataLoader(data, batch_size=2, shuffle=True, num_workers=4, pin_memory=True)
epoch_count = 0
epoch_loss = 0.
model = HRNet().train().cuda()

pretrained_dict = torch.load('/home/z/下载/work1/zmy1/hrnet_w48-8ef0771d.pth')
model_dict = model.state_dict()
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

loss_fn = nn.L1Loss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
print(get_parameter_number(model))

for epoch in range(args.num_epoch):
    start = time()
    epoch_loss = 0
    for i_batch, sample_batch in enumerate(train_loader):
        input = sample_batch[0].cuda()
        gt = sample_batch[1].cuda()
        out = model(input)
        gt0 = F.interpolate(gt, size=(120, 160), mode="bilinear", align_corners=True)
        loss = loss_fn(out, gt0)
        epoch_loss += float(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i_batch % 100 == 0:
            print("loss:{}".format(float(loss)))
    print('+++++++++++++++epoch loss:{}, time:{:.2f} min'.format((float(epoch_loss)/5000), (time()-start)/60))
    epoch_count += 1
    print("epoch_count        {}".format(epoch_count))
    if epoch % args.save_freq == 0:
        torch.save(model.state_dict(), './rms/my 7.26 epoch'+str(epoch_count)+'.pth')