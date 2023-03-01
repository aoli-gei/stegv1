import torch
from torch import nn

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
                PreNorm(dim, Attention(dim, heads=heads,
                        dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(
                    dim, mlp_dim, dropout=dropout))    # mlp
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class model1_encoder(nn.Module):
    def __init__(self, *, image_size, patch_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
            image_size: 输入图像大小
            patch_size: patch size
            patch_dim: 每一个 patch 的维度，也就是按块展开后的长度 (pw*ph*c)
            dim: 进行嵌入后的维度
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
        patch_dim = patch_height*patch_width*3
        self.to_patch_embedding = nn.Sequential(    # 输入进行线性嵌入，似乎有点不一样？
            Rearrange('b n c (h p1) (w p2) -> b n (h w) (p1 p2 c)', n=2, p1=patch_height, p2=patch_width),  # 转为 Token
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches, dim))  # 位置嵌入是可学习的
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.feature_reconstruct = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=3, stride=1, padding=1)  # 重建图像

        self.to_latent = nn.Identity()    # 这是啥

    def forward(self, img):
        # extract_feature = self.feature_extractor(img)不再使用 CNN 提取特征
        x = rearrange(img, 'b (n c) h w -> b n c h w', n=2)

        x = self.to_patch_embedding(x)    # (B,2,hw,dim)
        b, _, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.dropout(x)
        x = rearrange(x, 'b n hw l -> b (n hw) l', n=2)  # 组成长度为两倍的 patch 一起输入

        x = self.transformer(x)  # (B,H'xW',dim)

        x = rearrange(x, 'b (n h w) (p1 p2 c) -> b (n c) (h p1) (w p2)', n=2, h=self.proportion,
                      p1=self.patch_size, p2=self.patch_size)  # 转化为图像表示，(B,C,H,W)
        x = self.feature_reconstruct(x)

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
