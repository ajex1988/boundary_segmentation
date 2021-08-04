from torch import nn
import torchvision.models as models


class SingleObjRegressor(nn.Module):
    def __init__(self, sample_nums=16):
        super(SingleObjRegressor, self).__init__()
        self.backbone = models.googlenet(pretrained=True)
        self.sample_nums = sample_nums

    def forward(self, x):
        feature = self.backbone(x)
        fc_center = nn.Linear(1000,2)
        center = fc_center(feature)
        fc_coordinate = nn.Linear(1000,self.sample_nums)
        coordinate = fc_coordinate(feature)
        return center, coordinate
