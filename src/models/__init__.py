import torch.nn as nn
from torchvision import models as tvmodels


def _get_classification_model(model_meta):
    name = model_meta.name
    n_classes = model_meta.n_classes
    pretrained = model_meta.pretrained

    model = None
    if name == "resnet18":
        model = tvmodels.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif name == "resnet34":
        model = tvmodels.resnet34(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif name == "resnet50":
        model = tvmodels.resnet50(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
    elif name == "resnet101":
        model = tvmodels.resnet101(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, n_classes)
        
    return model
