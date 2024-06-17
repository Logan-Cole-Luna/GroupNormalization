import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p, d):
    return (k - 1) // 2 * d

class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, norm_type="none", fuse_norm=False):
        super().__init__()
        self.p = autopad(k, p, d) if p is None else p
        self.conv = nn.Conv2d(c1, c2, k, s, self.p, groups=g, dilation=d, bias=False)
        self.norm_type = norm_type
        self.fuse_norm = fuse_norm

        if norm_type == "group":
            num_groups = max(1, int(c2 / 2))
            self.bn = nn.GroupNorm(num_groups, c2)
            if fuse_norm:
                self.fuse_group_norm()
        elif norm_type == "batch":
            self.bn = nn.BatchNorm2d(c2)
        elif norm_type == "none":
            self.bn = None
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        print("Weight shape at forward:", self.conv.weight.shape)
        if self.conv.bias is not None:
            print("Bias shape at forward:", self.conv.bias.shape)
        else:
            print("Bias is None")

        if self.fuse_norm and self.norm_type == "group":
            return self.act(self.conv(x))
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)

    def fuse_group_norm(self):
        """ Fuse the GroupNorm parameters with Conv weights for inference optimization """
        with torch.no_grad():
            weight = self.conv.weight
            bias = torch.zeros(self.conv.out_channels, device=weight.device) if self.conv.bias is None else self.conv.bias

            gamma = self.bn.weight.view(self.conv.out_channels, 1, 1, 1)
            beta = self.bn.bias.view(self.conv.out_channels)
            mean = torch.zeros(self.conv.out_channels, device=weight.device)
            var = torch.ones(self.conv.out_channels, device=weight.device)

            print(f"Initial weight: {weight}")
            print(f"Initial bias: {bias}")
            print(f"Gamma: {gamma}")
            print(f"Beta: {beta}")
            print(f"Mean: {mean}")
            print(f"Var: {var}")

            new_weight = weight * (gamma / torch.sqrt(var.view(-1, 1, 1, 1) + self.bn.eps))
            new_bias = beta + (bias - mean * gamma.squeeze() / torch.sqrt(var + self.bn.eps))

            self.conv.weight = nn.Parameter(new_weight)
            self.conv.bias = nn.Parameter(new_bias.view(-1))
            self.bn = None

            print("New weight shape:", self.conv.weight.shape)
            print("New bias shape:", self.conv.bias.shape)

# Example usage
N, C, H, W = 1, 8, 32, 32
x = torch.randn(N, C, H, W)

model = Conv(C, C, 3, norm_type="group", fuse_norm=True)
print("Weight shape before forward pass:", model.conv.weight.shape)
print("Bias shape before forward pass:", model.conv.bias.shape)
output_fused = model(x)


# Testing the module without fusing GroupNorm
model = Conv(C, C, 3, norm_type="group", fuse_norm=False)
output_normal = model(x)

# Verify outputs
print("Output with fused GroupNorm:\n", output_fused)
print("Output with separate GroupNorm:\n", output_normal)
print("Difference between the outputs (should be near zero if fusion is correct):\n", (output_fused - output_normal).abs().max())
