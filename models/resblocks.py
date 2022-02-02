import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit


class ResBlockDownsample(jit.ScriptModule):
    def __init__(self, in_dim, out_dim, filter_size):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=filter_size, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=filter_size, padding=1, stride=1)
        self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, stride=1)

    @jit.script_method
    def forward(self, x):
        short = self.shortcut(F.avg_pool2d(x, 2))
        out = self.conv1(torch.relu(self.bn1(x)))
        out = F.avg_pool2d(self.conv2(torch.relu(self.bn2(out))), 2)

        return out + short


class ResBlockUpsample(jit.ScriptModule):
    def __init__(self, in_dim, out_dim, filter_size, norm_inputs=False):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim, in_dim, kernel_size=filter_size, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.conv2 = nn.Conv2d(in_dim, out_dim, kernel_size=filter_size, padding=1, stride=1)
        self.shortcut = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, stride=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.norm_inputs = norm_inputs

    @jit.script_method
    def forward(self, x):
        short = self.shortcut(self.upsample(x))

        if not self.norm_inputs:
            x = torch.relu(self.bn1(x))
        out = self.upsample(self.conv1(x))
        out = self.conv2(torch.relu(self.bn2(out)))

        return out + short


class ResEncoder(jit.ScriptModule):
    def __init__(self, in_channels, dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.res1 = ResBlockDownsample(dim, 2 * dim, 3)
        self.res2 = ResBlockDownsample(2 * dim, 4 * dim, 3)
        self.res3 = ResBlockDownsample(4 * dim, 8 * dim, 3)
        self.res4 = ResBlockDownsample(8 * dim, 8 * dim, 3)

    @jit.script_method
    def forward(self, x):
        out = self.conv1(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        return out


class ResDecoder(jit.ScriptModule):
    def __init__(self, out_channels, dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1)
        self.res1 = ResBlockUpsample(8 * dim, 8 * dim, 3, norm_inputs=True)
        self.res2 = ResBlockUpsample(8 * dim, 4 * dim, 3)
        self.res3 = ResBlockUpsample(4 * dim, 2 * dim, 3)
        self.res4 = ResBlockUpsample(2 * dim, dim, 3)
        self.bn4 = nn.BatchNorm2d(dim)

    @jit.script_method
    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        return out


if __name__ == '__main__':
    e = ResEncoder(3, 64)
    d = ResDecoder(3, 64)
    a = torch.rand(1, 3, 64, 64)
    print((e(a)).shape)
