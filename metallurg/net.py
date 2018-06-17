from torch.nn import Module, Conv2d, Linear, ReLU, MaxPool2d, Dropout
from torch.nn import BatchNorm1d, BatchNorm2d


def conv3x3(in_, out):
    return Conv2d(in_, out, 3, padding=1)


class ConvBNRelu(Module):

    def __init__(self, in_, out):
        super(ConvBNRelu, self).__init__()

        self.conv = conv3x3(in_, out)
        self.bn = BatchNorm2d(out)
        self.activation = ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class DenseBNRelu(Module):

    def __init__(self, input, output):
        super(DenseBNRelu, self).__init__()

        self.fc = Linear(input, output)
        self.bn = BatchNorm1d(output)
        self.activation = ReLU(inplace=True)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.activation(x)

        return x


class MetallurgNet(Module):

    def __init__(self):
        super(MetallurgNet, self).__init__()

        self.pool = MaxPool2d(2, 2)
        self.dropout = Dropout(inplace=True)

        self.conv1 = ConvBNRelu(1, 6)
        self.conv2 = ConvBNRelu(6, 12)

        self.fc1 = DenseBNRelu(12 * 16 * 16, 1024)
        self.fc2 = DenseBNRelu(1024, 128)
        self.fc3 = Linear(128, 12)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))

        x = x.view(-1, 12 * 16 * 16)

        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.fc3(x)

        return x


def get_metallurg_net(filename):
    import torch

    model = MetallurgNet()
    model.load_state_dict(torch.load(filename))

    return model.eval()
