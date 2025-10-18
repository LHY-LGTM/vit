import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224,patch_size=16,in_c=3,embed_dim=768,norm_layer=None):
        """
            img_size 图像大小
            patch size 每个patch的大小
        """
        super().__init__()
        img_size=(img_size,img_size)
        patch_size=(patch_size,patch_size)
        self.img_size=img_size
        self.patch_size=patch_size
        self.grid_size=(img_size[0]//patch_size[0],img_size[1]//patch_size[1])#patch 的网格大小
        self.num_patchs=self.grid_size[0]*self.grid_size[1] #patch 的总数

        self.proj=nn.Conv2d(in_c,embed_dim,kernel_size=patch_size,stride=patch_size)# 3,224,224 ->768,14,14
        self.norm=norm_layer(embed_dim) if norm_layer else nn.Identity() #如果有的话则使用，没有的话则默认保持不变


    def forward(self,x):
        B,C,H,W=x.shape #获取我们输入张量的形状
        assert H==self.img_size[0] and W==self.img_size[1],\
        f"输入图像的大小{H}*{W}与模型期望大小{self.img_size[0]}*{self.img_size[1]}不匹配"
        # B,3,224,224 ->B,768,14,14 ->B,768,14*14 ->B, 14*14,768
        x=self.proj(x).flatten(2).transpose(1,2)
        x=self.norm(x) #如有归一化层，则使用
        return x

