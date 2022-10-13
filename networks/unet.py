

from typing import Tuple, Union
import torch
import torch.nn as nn

from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock

def conv_block_3d(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        )

def conv_trans_block_3d(in_dim, out_dim):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        )

def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)

def conv_block_2_3d(in_dim, out_dim):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)


class UNET(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
     
    ) -> None:
        
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = 12
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size,
            out_channels=feature_size * 2,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
       
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size * 4,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
      
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 8,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore
        self.pool = max_pooling_3d()
        self.bridge = conv_block_2_3d(feature_size * 8, feature_size * 16)
        self.trans_1 = conv_trans_block_3d(feature_size * 16, feature_size * 16)
        self.up_1 = conv_block_2_3d(384, feature_size * 8)
        self.trans_2 = conv_trans_block_3d(feature_size * 8, feature_size * 8)
        self.up_2 = conv_block_2_3d(feature_size * 12, feature_size * 4)
        self.trans_3 = conv_trans_block_3d(feature_size * 4, feature_size * 4)
        self.up_3 = conv_block_2_3d(feature_size * 6, feature_size * 2)
        self.trans_4 = conv_trans_block_3d(feature_size * 2, feature_size * 2)
        self.up_4 = conv_block_2_3d(feature_size * 3, feature_size * 1)
        

   
    def forward(self, x_in):
        
        enc1 = self.encoder1(x_in)
       
        pool1 = self.pool(enc1)
       
        enc2 = self.encoder2(pool1)
        
        pool2 = self.pool(enc2)

        enc3 = self.encoder3(pool2)
        
        pool3 = self.pool(enc3)

        enc4 = self.encoder4(pool3)
        
        pool4 = self.pool(enc4)
        
        bridge = self.bridge(pool4)
        
        trans_1 = self.trans_1(bridge)
        
        concat_1 = torch.cat([trans_1, enc4], dim=1)
        
        up_1 = self.up_1(concat_1) 
        
        trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
        concat_2 = torch.cat([trans_2, enc3], dim=1) # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        
        trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
        concat_3 = torch.cat([trans_3, enc2], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        
        trans_4 = self.trans_4(up_3) # -> [1, 16, 64, 64, 64]
        concat_4 = torch.cat([trans_4, enc1], dim=1) # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4)
        
        logits = self.out(up_4)
        
        return logits
