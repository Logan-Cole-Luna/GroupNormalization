import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the fusion function for Group Norm and Conv layers
def fuse_group_norm_conv(conv, gn):
    # Ensure the layers are compatible for fusion
    assert isinstance(conv, nn.Conv2d) and isinstance(gn, nn.GroupNorm)

    # Get the convolution parameters
    conv_weight = conv.weight
    conv_bias = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels).to(conv.weight.device)

    # Get the group normalization parameters
    gn_weight = gn.weight
    gn_bias = gn.bias
    eps = gn.eps

    # Calculate the number of channels per group
    C = conv.out_channels
    G = gn.num_groups
    C_per_group = C // G

    # Ensure no division by zero
    if C_per_group == 0:
        raise ValueError("The number of channels per group cannot be zero.")

    # Compute the scale and shift
    scale = gn_weight.view(G, C_per_group, 1, 1).repeat(1, 1, 1, 1).view(C, 1, 1, 1)
    shift = gn_bias.view(G, C_per_group, 1, 1).repeat(1, 1, 1, 1).view(C)

    # Fuse the convolution weights and biases
    fused_weight = conv_weight * scale
    fused_bias = conv_bias * scale.view(C) + shift

    # Create a new convolution layer with the fused parameters
    fused_conv = nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels, kernel_size=conv.kernel_size,
                           stride=conv.stride, padding=conv.padding, dilation=conv.dilation, groups=conv.groups,
                           bias=True)

    fused_conv.weight.data = fused_weight
    fused_conv.bias.data = fused_bias

    return fused_conv


# Define a sample neural network with Conv2d and GroupNorm
class SampleNet(nn.Module):
    def __init__(self):
        super(SampleNet, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.gn = nn.GroupNorm(num_groups=4, num_channels=16)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        return x


# Create a function to test the fusion
def test_fusion():
    # Create a sample input
    x = torch.randn(1, 3, 32, 32)

    # Initialize the network and get the output before fusion
    model = ConvNet()
    before_fusion_output = model(x)

    # Fuse the GroupNorm and Conv2d layers
    fused_conv = fuse_group_norm_conv(model.conv, model.gn)

    # Replace the layers in the model with the fused layer
    model.conv = fused_conv
    model.gn = nn.Identity()  # Remove GroupNorm layer

    # Get the output after fusion
    after_fusion_output = model(x)

    # Compare the outputs
    print("Difference after fusion:", torch.abs(before_fusion_output - after_fusion_output).max().item())


# Updated ConvNet model with GroupNorm and fusion functionality
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(num_groups=8, num_channels=32)
        self.fc1 = nn.Linear(in_features=32 * 8 * 8, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2(x)
        x = self.gn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x



# Test fusion function
if __name__ == "__main__":
    # Initialize the model, loss function, and optimizer
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Dummy input and output for testing
    input_tensor = torch.randn(1, 3, 32, 32)
    output_tensor = torch.tensor(1)  # Corrected target tensor

    # Forward pass
    output = model(input_tensor)
    loss = criterion(output, output_tensor.unsqueeze(0))  # Ensuring correct shape

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("Output:", output)
    print("Loss:", loss.item())

    test_fusion()
