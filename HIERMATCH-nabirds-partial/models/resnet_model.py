import torch
import torch.nn as nn
import torchvision
from torchvision import models


class Resnet18Fc(nn.Module):
  def __init__(self, prt_flag = True):
    super(Resnet18Fc, self).__init__()
    model_resnet18 = models.resnet18(pretrained = prt_flag)
    model_resnet18.fc = torch.nn.Identity()
    self.model_resnet18 = model_resnet18

  def forward(self, x):
    return self.model_resnet18(x)


class Resnet50Fc(nn.Module):
  def __init__(self, prt_flag = True):
    super(Resnet50Fc, self).__init__()
    model_resnet50 = models.resnet50(pretrained = prt_flag)
    model_resnet50.fc = torch.nn.Identity()
    self.model_resnet50 = model_resnet50

  def forward(self, x):
    return self.model_resnet50(x)


class WResnet50Fc(nn.Module):
  def __init__(self, prt_flag = True):
    super(WResnet50Fc, self).__init__()
    model_wresnet50 = models.wide_resnet50_2(pretrained = prt_flag)
    model_wresnet50.fc = torch.nn.Identity()
    self.model_wresnet50 = model_wresnet50

  def forward(self, x):
    return self.model_wresnet50(x)


class WResnet101Fc(nn.Module):
  def __init__(self, prt_flag = True):
    super(WResnet101Fc, self).__init__()
    model_wresnet101 = models.wide_resnet101_2(pretrained = prt_flag)
    model_wresnet101.fc = torch.nn.Identity()
    self.model_wresnet101 = model_wresnet101

  def forward(self, x):
    return self.model_wresnet101(x)


class model_bn_2(nn.Module):
    def __init__(self, model, feature_size=512, classes=[555, 404]):

        super(model_bn_2, self).__init__() 

        self.features_2 =  model
        
        val = feature_size // 2
        self.feature_division = [val, feature_size]
        
        self.feature_size = feature_size

        self.classifier_1 = nn.Linear(feature_size, 404)
        self.classifier_2 = nn.Linear(feature_size - val, 555)
 
    def forward(self, x):

        x = self.features_2(x)
        x_2 =  x[:, : self.feature_division[0]]
        x_3 =  x[:, self.feature_division[0] : ]

        family_input = torch.cat([     x_2, x_3.detach()], 1)
        species_input = x_3
        
        family_out = self.classifier_1(family_input)
        species_out = self.classifier_2(species_input)

        return [species_out, family_out]


class model_bn_3(nn.Module):
    def __init__(self, model, feature_size=512, classes=[555, 404, 50]):

        super(model_bn_3, self).__init__() 
        self.features_2 =  model
        
        val = feature_size // 3
        self.feature_division = [val, 2 * val, feature_size]
        
        self.feature_size = feature_size

        self.classifier_1 = nn.Linear(feature_size , classes[2])
        self.classifier_2 = nn.Linear(feature_size - val, classes[1])
        self.classifier_3 = nn.Linear(feature_size - 2 * val, classes[0])
 
    def forward(self, x):

        x = self.features_2(x)
        x_1 =  x[:,   : self.feature_division[0]]
        x_2 =  x[:, self.feature_division[0] : self.feature_division[1]]
        x_3 =  x[:, self.feature_division[1] : ]

        order_input  = torch.cat([x_1, x_2.detach(), x_3.detach()], 1)
        family_input = torch.cat([     x_2, x_3.detach()], 1)
        species_input = x_3
        
        order_out = self.classifier_1(order_input)
        family_out = self.classifier_2(family_input)
        species_out = self.classifier_3(species_input)

        return [species_out, family_out, order_out] # from fine to coarse
