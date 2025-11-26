"""Model definitions including backbone, classifier, and domain classifier"""
import torch
import torch.nn as nn
from torchvision import models


class GradReverse(torch.autograd.Function):
    """Gradient Reversal Layer for DANN"""
    
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd=1.0):
    """Apply gradient reversal"""
    return GradReverse.apply(x, lambd)


class DomainClassifier(nn.Module):
    """Domain classifier for DANN"""
    
    def __init__(self, in_features, hidden=256, num_domains=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden//2, num_domains)
        )
    
    def forward(self, x):
        return self.net(x)


class ClassifierHead(nn.Module):
    """Classification head for multiclass classification"""
    
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.fc(x)


class BackboneWrapper(nn.Module):
    """Wrapper for backbone model"""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)


def build_backbone(name='resnet18', pretrained=True):
    """Build a backbone network"""
    if name == 'resnet18':
        m = models.resnet18(pretrained=pretrained)
    elif name == 'resnet50':
        m = models.resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError(f'Backbone {name} not supported')
    
    in_feat = m.fc.in_features
    m.fc = nn.Identity()
    
    return BackboneWrapper(m), in_feat