import torch
import torch.nn as nn
import torch.optim
import math
import numpy as np
from critic import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
from thop import profile
import torch.nn.functional as F

# 以类的方式定义参数


class Args:
    def __init__(self) -> None:
        self.batch_size = 8
        self.image_size = 256
        self.patch_size = 16
        self.lr = 2e-4
        self.epochs = 6000
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.val_freq = 2
        self.save_freq = 50
        self.train_next = 0


args = Args()

# 设置随机种子
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 损失函数
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6
 
    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss

def steg_loss(img1, img2):
    # loss_fn = torch.nn.MSELoss(reduce=True, size_average=False) # L2 损失
    loss_fn=L1_Charbonnier_loss()   # Charbonnier 损失
    loss = loss_fn(img1, img2)
    return loss.to(args.device)


def reconstruction_loss(img1, img2):
    # loss_fn = torch.nn.MSELoss(reduce=True, size_average=False)
    loss_fn=L1_Charbonnier_loss()   # Charbonnier 损失
    loss = loss_fn(img1, img2)
    return loss.to(args.device)


# tensorboard
writer = SummaryWriter('/content/drive/MyDrive/stegv1/runs')

# 模型初始化
encoder = model1_encoder(image_size=256, patch_size=16, dim=2048, extract_dim=8, depth=12, heads=12, mlp_dim=3072)   # dim 是 token 维度，dim=(patch_height x patch_width x channel)
decoder = model1_decoder(image_size=256, patch_size=16, dim=1024, depth=1, heads=12, mlp_dim=3072)
encoder.cuda()
decoder.cuda()

# 计算模型参数量
with torch.no_grad():
    test_encoder_input = torch.randn(1,6,256,256).cuda()
    test_decoder_input = torch.randn(1,3,256,256).cuda()
    encoder_mac,encoder_params=profile(encoder,inputs=(test_encoder_input,))
    decoder_mac,decoder_params=profile(decoder,inputs=(test_decoder_input,))
    print("thop result:encoder FLOPs="+str(encoder_mac*2)+",encoder params="+str(encoder_params))
    print("thop result:decoder FLOPs="+str(decoder_mac*2)+",decoder params="+str(decoder_params))

# 优化器以及学习率衰减
optim = torch.optim.AdamW([{'params': encoder.parameters()}, {'params': decoder.parameters()}], lr=args.lr)
scheduler =torch.optim.lr_scheduler.StepLR(optim,step_size=1000,gamma=0.75)


# 损失函数
s_loss=L1_Charbonnier_loss().to(args.device)
r_loss=L1_Charbonnier_loss().to(args.device)

# 加载模型
model_path=''
if args.train_next!=0:
  model_path='/content/drive/MyDrive/stegv1/model/model_checkpoint_%.5i'%args.train_next +'.pt'
  state_dicts = torch.load(model_path)
  encoder.load_state_dict(state_dicts['encoder'],strict=True)
  decoder.load_state_dict(state_dicts['decoder'],strict=True)
  optim.load_state_dict(state_dicts['opt'],strict=True)

# train
for i_epoch in range(args.epochs):
    sum_loss = []
    for i_batch, (cover, secret) in enumerate(zip(DIV2K_train_cover_loader, DIV2K_train_secret_loader)):
        cover = cover.to(args.device)
        secret = secret.to(args.device)
        input_img = torch.cat((cover, secret), 1)
        input_img = input_img.to(args.device)

        # encode
        encode_img = F.sigmoid(cover + encoder(input_img)) # 添加残差连接

        # decode
        decode_img = F.sigmoid(decoder(encode_img))

        # loss
        hide_loss = s_loss(cover.cuda(), encode_img.cuda())
        rev_loss = r_loss(secret.cuda(), decode_img.cuda())

        # 先训练编码器再训练解码器
        # if i_epoch <100:
        #     total_loss=h_loss
        # else :
        total_loss = hide_loss+rev_loss
        sum_loss.append(total_loss.item())
        print("total loss: "+str(total_loss.item())+" hide_loss: "+str(hide_loss.item())+" rev_loss: "+str(rev_loss.item()))

        # backward
        total_loss.backward()
        optim.step()
        optim.zero_grad()

    # 进行验证，并记录指标
    if i_epoch % args.val_freq == 0 and i_epoch!=0:
        print("validation begin:")
        with torch.no_grad():
            # val
            encoder.eval()
            decoder.eval()

            # 评价指标
            psnr_secret = []
            psnr_cover = []
            ssim_secret = []
            ssim_cover = []

            # 在验证集上测试
            for (cover, secret) in zip(DIV2K_val_cover_loader,DIV2K_val_secret_loader):
                cover = cover.to(args.device)
                secret = secret.to(args.device)
                input_img = torch.cat((cover, secret), 1)
                input_img = input_img.to(args.device)

                # encode
                encode_img = F.sigmoid(cover + encoder(input_img))

                # decode
                decode_img = F.sigmoid(decoder(encode_img))

                # 计算各种指标
                # 拷贝进内存以方便计算
                cover = cover.cpu()
                secret = secret.cpu()
                encode_img = encode_img.cpu()
                decode_img = decode_img.cpu()

                psnr_encode_temp = calculate_psnr(cover, encode_img)
                psnr_decode_temp = calculate_psnr(secret, decode_img)
                psnr_cover.append(psnr_encode_temp)
                psnr_secret.append(psnr_decode_temp)

                ssim_encode_temp = calculate_ssim(cover, encode_img)
                ssim_decode_temp = calculate_ssim(secret, decode_img)
                ssim_cover.append(ssim_encode_temp)
                ssim_secret.append(ssim_decode_temp)
                writer.add_images('image/encode_img',encode_img,dataformats='NCHW',global_step=i_epoch)
                writer.add_images('image/decode_img',decode_img,dataformats='NCHW',global_step=i_epoch)

            # 写入 tensorboard
            writer.add_scalar("PSNR/PSNR_cover", np.mean(psnr_cover), i_epoch)
            writer.add_scalar("PSNR/PSNR_secret", np.mean(psnr_secret), i_epoch)
            writer.add_scalar("SSIM/SSIM_cover", np.mean(ssim_cover), i_epoch)
            writer.add_scalar("SSIM/SSIM_secret", np.mean(ssim_secret), i_epoch)
            print("PSNR_cover:" + str(np.mean(psnr_cover)) + " PSNR_secret:" + str(np.mean(psnr_secret)))
            print("SSIM_cover:" + str(np.mean(ssim_cover)) + " SSIM_secret:" + str(np.mean(ssim_secret)))

    # 绘制损失函数曲线
    print("epoch:"+str(i_epoch) + ":" + str(np.mean(sum_loss)))

    # 保存当前模型以及优化器参数
    if (i_epoch % args.save_freq) == 0:
        torch.save({'opt': optim.state_dict(),
                    'encoder': encoder.state_dict(),
                    'decoder': decoder.state_dict()}, '/content/drive/MyDrive/stegv1/model/model_checkpoint_%.5i' % i_epoch+'.pt')

torch.save({'opt': optim.state_dict(),
            'encoder': encoder.state_dict(),
            'decoder': decoder.state_dict()}, 'stegv1/model/model_last.pt')
