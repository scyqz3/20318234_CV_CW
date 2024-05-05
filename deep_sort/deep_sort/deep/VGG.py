import torch.nn as nn
import torchvision.models as models

class VGG(nn.Module):
    def __init__(self, num_classes=1000, reid=False):
        super().__init__()
        self.reid = reid

        # Load the pre-trained VGG16 model
        self.vgg = models.vgg16(pretrained=True)

        self.vgg.classifier[-1] = nn.Linear(self.vgg.classifier[-1].in_features, num_classes)

        # Keep the convolution layer part
        self.vgg_conv = nn.Sequential(*list(self.vgg.features.children()))

    def forward(self, x):
        features = self.vgg_conv(x)
        features = features.view(features.size(0), -1)

        if self.reid:
            # L2 normalization of the feature vector
            features = features.div(features.norm(p=2, dim=1, keepdim=True))
            return features

        # Category output
        return self.vgg.classifier(features)
