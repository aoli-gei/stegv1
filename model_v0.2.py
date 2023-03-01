import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes
# LayerNorm


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# Feed Forward(MLP)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# self attention


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# transformer block


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])  # 存储 Transformer 的每一个块
        for _ in range(depth):  # 看要堆叠多少个 Transformer blcok
            self.layers.append(nn.ModuleList([
                # self attention
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))    # mlp
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class SE_block(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.FC = nn.Linear(in_channels, in_channels)

    def forward(self, img):
        b, c, _, _ = img.shape
        sequeeze = F.adaptive_avg_pool2d(img, (b, c))
        attention_weight = F.sigmoid(self.FC(sequeeze))
        attention_out = attention_weight*img
        return attention_out


class bottleneck_block(nn.Module):
    """
        感觉该模型通道数不是很多，直接提升通道
    """

    def __init__(self, in_channels, bottle_dim):
        super().__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=bottle_dim, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=bottle_dim, out_channels=bottle_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=bottle_dim, out_channels=in_channels, kernel_size=1, stride=0, padding=0)
        )

    def forward(self, img):
        x = self.bottleneck(img)
        return img+x


class stg_block(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
            最小模块，由 Transformer 分支以及 CNN 分支组成，输入为激活图 (B,C,H,W)，CNN 分支使用 SENet 做通道注意力
            image_size: 输入图像大小
            patch_size: patch size
            dim: token's length
            heads: numbers of multi-attention head
            mlp_dim: mlp's dim
            dim_head: one of the head's dim
            channels:每张图片的通道数！！！
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height//patch_height)*(image_width//patch_width)
        self.proportion = image_height//patch_height  # 记录 patch 大小和图像大小之间的比例，以便后面进行转换
        self.patch_size = patch_height

        self.transformer_block = nn.Sequential(
            PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
            PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
        )
        self.cnn_blcok = nn.Sequential(
            nn.Conv2d(in_channels=2*channels, out_channels=4*channels, kernel_size=3, stride=2, padding=1),  # 特征通道数翻倍，分辨率减半
            nn.BatchNorm2d(num_features=4*channels),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=4*channels, out_channels=2*channels, kernel_size=3, stride=2, padding=1, output_padding=1),   # 恢复分辨率
            nn.ReLU(),
        )

    def forward(self, img):
        img_token = rearrange(img, 'b (n c) (h p1) (w p2) -> b (n h w) (p1 p2 c)', n=2, p1=self.patch_size,
                              p2=self.patch_size)  # 转为 Token（B,2N,L)n:图像数量，h：纵向patch数量，w：横向patch数量，c：通道数
        transformer_output = self.transformer_block(img_token)    # Transformer输出
        transformer_output = rearrange(transformer_output, 'b (n h w) (p1 p2 c) -> b (n c) (h p1) (w p2)', n=2,
                                       h=self.proportion, p1=self.patch_size, p2=self.patch_size)    # 转为图像表示
        cnn_output = self.cnn_blcok(img)    # CNN 输出
        res_output = cnn_output+transformer_output+img    # 残差连接
        return res_output


class UNet_block(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(2):
            self.layers.add_module(stg_block(image_size=image_size, patch_size=patch_size, dim=dim,
                                   heads=heads, mlp_dim=mlp_dim, channels=channels, dim_head=dim_head))

    def forward(self, img):
        for layer in self.layers:
            img = layer(img)
        return img


class UNet(nn.Module):
    def __init(self, *, image_size, patch_size, dim, heads, mlp_dim, in_channel, out_channel=3, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
            构造 U-Net 网络
            image_size: 输入图像大小
            patch_size: patch size
            dim: token's length
            heads: numbers of multi-attention head
            mlp_dim: mlp's dim
            dim_head: one of the head's dim
            channels:每张图片的通道数！！！ 
        """
        super().__init__()
        filter = [in_channel, in_channel*2, in_channel*4]
        self.down_conv1 = nn.Conv2d(in_channels=filter[0], out_channels=filter[1], kernel_size=3, stride=2, padding=1)
        self.down_conv2 = nn.Conv2d(in_channels=filter[1], out_channels=filter[2], kernel_size=3, stride=2, padding=1)
        self.up_conv1 = nn.ConvTranspose2d(in_channels=filter[3], out_channels=filter[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up_conv2 = nn.ConvTranspose2d(in_channels=filter[2], out_channels=filter[1], kernel_size=3, stride=2, padding=1, output_padding=1)


        self.block1=UNet_block(image_size=image_size,patch_size=8,dim=8**2*filter[0],heads=8,mlp_dim=8**2*filter[0]*2,dim_head=8*filter[0]*2)
        self.block2=UNet_block(image_size=image_size,patch_size=8,dim=8**2*filter[0],heads=8,mlp_dim=8**2*filter[0]*2,dim_head=8*filter[0]*2)
        # 写到这里


class model1_encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, extract_dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
            image_size: 输入图像大小
            patch_size: patch size
            dim: token's length
            extract_dim: feature extractor output's dim
            depth: numbers of transformer block
            heads: numbers of multi-attention head
            mlp_dim: mlp's dim
            dim_head: one of the head's dim
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'  # 防止划分不完全

        num_patches = (image_height//patch_height)*(image_width//patch_width)
        self.proportion = image_height//patch_height  # 记录 patch 大小和图像大小之间的比例，以便后面进行转换
        self.patch_size = patch_height
        # 每个 patch 看成是一个 Token，其维度为 H'xW'xC
        patch_dim = extract_dim*patch_height*patch_width
        self.to_patch_embedding = nn.Sequential(    # 输入进行线性嵌入，似乎有点不一样？
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),  # 转为 Token
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))  # 位置嵌入是可学习的
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.feature_extractor = nn.Conv2d(in_channels=channels*2, out_channels=extract_dim, kernel_size=3, stride=1, padding=1)  # 使用 CNN 初步提取特征
        self.to_latent = nn.Identity()    # 这是啥

    def forward(self, img):
        extract_feature = self.feature_extractor(img)
        x = self.to_patch_embedding(extract_feature)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)  # (B,H'xW',dim)

        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.proportion,
                      p1=self.patch_size, p2=self.patch_size)  # 转化为图像表示，(B,C,H,W)

        return x


class model1_decoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth=1, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
            image_size: 输入图像大小
            patch_size: patch size
            dim: token's length
            extract_dim: feature extractor output's dim
            depth: numbers of transformer block
            heads: numbers of multi-attention head
            mlp_dim: mlp's dim
            dim_head: one of the head's dim
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'  # 防止划分不完全

        num_patches = (image_height//patch_height)*(image_width//patch_width)
        self.proportion = image_height//patch_height  # 记录 patch 大小和图像大小之间的比例，以便后面进行转换
        self.patch_size = patch_height
        # 每个 patch 看成是一个 Token，其维度为 H'xW'xC
        patch_dim = channels*patch_height*patch_width
        self.to_patch_embedding = nn.Sequential(    # 输入进行线性嵌入，似乎有点不一样？
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),  # 转为 Token
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, dim))  # 位置嵌入是可学习的？
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.to_latent = nn.Identity()    # 这是啥

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x)  # (B,H'xW',dim)

        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.proportion,
                      p1=self.patch_size, p2=self.patch_size)  # 转化为图像表示，(B,C,H,W)

        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),   # 转为 Token
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
