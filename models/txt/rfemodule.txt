import torch
import torch.nn as nn
import torch.nn.functional as F

class RFEModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 3, 5]):
        super(RFEModule, self).__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            for d in dilation_rates
        ])
        self.conv1x1 = nn.Conv2d(len(dilation_rates) * out_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        branch_outs = [branch(x) for branch in self.branches]
        out = torch.cat(branch_outs, dim=1)  # Concatenate along channel dimension
        out = self.conv1x1(out)
        out = self.bn(out)
        out = self.relu(out + x)  # Residual connection
        return out
