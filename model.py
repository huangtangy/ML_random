import torch as t
from torch import nn
from torch.nn import functional as F
class mymodel():
  # 2dCNN
  feat_map = 16
  drop_out = 0.1

  Conv = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=feat_map, kernel_size=7, stride=1, padding=1),# 第一个卷积层
      #输出通道 卷积核数量 等于feature map的数量
      nn.ReLU(inplace=True),#非线性激活层
      nn.Dropout(drop_out),
      nn.MaxPool2d(kernel_size=2),#池化大小2*2

      nn.Conv2d(in_channels=feat_map, out_channels=feat_map, kernel_size=7, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Dropout(drop_out),
      nn.MaxPool2d(kernel_size=2),

      nn.Conv2d(in_channels=feat_map, out_channels=feat_map, kernel_size=7, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Dropout(drop_out),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(in_channels=feat_map, out_channels=feat_map, kernel_size=7, stride=1, padding=1),

      nn.Flatten(),
      nn.Linear(2304, 50),#2304
      nn.ReLU(inplace=True),
      nn.Linear(50, 2)
  )
# 1d
  feat_map = 16
  drop_out = 0.1
  Conv1d = nn.Sequential(

      nn.Conv1d(in_channels=1, out_channels=feat_map, kernel_size=7, stride=1, padding=1),  # 第一个卷积层
      nn.ReLU(inplace=True),  # 非线性激活层
      nn.Dropout(drop_out),
      nn.MaxPool1d(kernel_size=2),  # 池化大小2*2

      nn.Conv1d(in_channels=feat_map, out_channels=feat_map, kernel_size=7, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Dropout(drop_out),
      nn.MaxPool1d(kernel_size=2),
      nn.Conv1d(in_channels=feat_map, out_channels=feat_map, kernel_size=7, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Dropout(drop_out),
      nn.MaxPool1d(kernel_size=2),
      nn.Conv1d(in_channels=feat_map, out_channels=feat_map, kernel_size=7, stride=1, padding=1),

      nn.Flatten(),
      nn.Linear(192, 50),
      nn.ReLU(inplace=True),
      nn.Linear(50, 2)
  )

class ResidualBlock(nn.Module):
    """
    实现子 module：Residual Block
    """
    def __init__(self, inchannel, outchannel, stride=1, shortcut=None):
        super(ResidualBlock, self).__init__()

        # 由于 Residual Block 分为左右两部分，因此定义左右两边的 layer
        # 定义左边
        self.left = nn.Sequential(
            # Conv2d 参数：in_channel,out_channel,kernel_size,stride,padding
            nn.Conv2d(inchannel, outchannel, 3, stride, 1, bias=False),
            #(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(outchannel)
            )
        # 定义右边
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)  # 检测右边直连的情况
        out += residual
        return F.relu(out)
class ResNet(nn.Module):
    """
    实现主 module：ResNet34
    ResNet34 包含多个 layer，每个 layer 又包含多个 residual block
    用子 module 实现 residual block，用 _make_layer 函数实现 layer
    """

    def __init__(self, num_classes=2):
        super(ResNet, self).__init__()
        # 前几层图像转换
        fm = 16
        self.pre = nn.Sequential(
            nn.Conv2d(1, fm, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fm), #nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3),#kernel_size, stride=None, padding
        )

        # 重复的 layer 分别有 3，4，6，3 个 residual block

        self.layer1 = self._make_layer(fm, fm*2, 3)
        self.layer2 = self._make_layer(fm*2, fm*4, 4, stride=2)
        self.layer3 = self._make_layer(fm*4, fm*8, 6, stride=2)
        self.layer4 = self._make_layer(fm*8, fm*8, 3, stride=2)

        # 分类用的全连接
        self.fc = nn.Linear(fm*8, num_classes)

    def _make_layer(self, inchannel, outchannel, block_num, stride=1):
        """
        构建 layer，包含多个 residual block
        """
        shortcut = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, 1, stride, bias=False),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(outchannel)
            )

        layers = []
        layers.append(ResidualBlock(inchannel, outchannel, stride, shortcut))

        for i in range(1, block_num):
            layers.append(ResidualBlock(outchannel, outchannel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)

        return self.fc(x)
