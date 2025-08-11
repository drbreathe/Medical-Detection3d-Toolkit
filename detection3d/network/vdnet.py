import torch
import torch.nn as nn

from detection3d.network.module.weight_init import kaiming_weight_init, gaussian_weight_init
from detection3d.network.module.vnet_inblock import InputBlock
from detection3d.network.module.vnet_outblock import OutputBlock
from detection3d.network.module.vnet_upblock import UpBlock
from detection3d.network.module.vnet_downblock import DownBlock


def parameters_kaiming_init(net):
    """ model parameters initialization """
    net.apply(kaiming_weight_init)


def parameters_gaussian_init(net):
    """ model parameters initialization """
    net.apply(gaussian_weight_init)


# class Net(nn.Module):
#     """ volumetric segmentation network """

#     def __init__(self, in_channels, out_channels):
#         super(Net, self).__init__()
#         self.in_block = InputBlock(in_channels, 16)
#         self.down_32 = DownBlock(16, 3, compression=False)
#         self.down_64 = DownBlock(32, 4, compression=False)
#         self.up_64 = UpBlock(64, 64, 4, compression=False)
#         self.up_32 = UpBlock(64, 32, 3, compression=False)
#         self.out_block = OutputBlock(32, out_channels)


#     def forward(self, input):
#         assert isinstance(input, torch.Tensor)

#         out16 = self.in_block(input)
#         out32 = self.down_32(out16)
#         out64 = self.down_64(out32)
#         out = self.up_64(out64, out32)
#         out = self.up_32(out, out16)
#         out = self.out_block(out)
#         return out

#     def max_stride(self):
#         return 4


class Net(nn.Module):
    """Deeper volumetric segmentation network"""

    def __init__(self, in_channels, out_channels):
        super(Net, self).__init__()

        self.in_block = InputBlock(in_channels, 16)

        self.down_32 = DownBlock(16, 2, compression=False)   # Output: 32 channels
        self.down_64 = DownBlock(32, 3, compression=False)   # Output: 64 channels
        self.down_128 = DownBlock(64, 4, compression=False)  # Output: 128 channels

        self.up_128 = UpBlock(128, 128, 4, compression=False)  # Skip from down_64
        self.up_64 = UpBlock(128, 64, 3, compression=False)    # Skip from down_32
        self.up_32 = UpBlock(64, 32, 2, compression=False)     # Skip from in_block

        self.out_block = OutputBlock(32, out_channels)

    def forward(self, input):
        assert isinstance(input, torch.Tensor)

        out16 = self.in_block(input)
        out32 = self.down_32(out16)
        out64 = self.down_64(out32)
        out128 = self.down_128(out64)

        up128 = self.up_128(out128, out64)
        up64 = self.up_64(up128, out32)
        up32 = self.up_32(up64, out16)

        out = self.out_block(up32)
        return out

    def max_stride(self):
        return 8  # You've added another 2x downsample level