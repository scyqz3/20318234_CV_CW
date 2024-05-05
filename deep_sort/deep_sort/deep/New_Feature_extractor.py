import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2


class VGG(nn.Module):
    def __init__(self, num_classes=1000, reid=False):
        super().__init__()
        self.reid = reid

        self.vgg = models.vgg16(pretrained=True)

        # Modify the fully connected network in VGG
        self.vgg.classifier[-1] = nn.Linear(self.vgg.classifier[-1].in_features, num_classes)

        # Keep only the convolutional part, remove the fully connected layers
        self.vgg_conv = nn.Sequential(*list(self.vgg.features.children()))

    def forward(self, x):
        features = self.vgg_conv(x)
        features = features.view(features.size(0), -1)

        if self.reid:
            # Perform L2 normalization on the feature vectors
            features = features.div(features.norm(p=2, dim=1, keepdim=True))
            return features

        return features



class FeatureExtractor(object):
    """
        Feature Extractor:
        Extract the features corresponding to the bounding box, and obtain a fixed-dimensional embedding as
        the representative of the bounding box, for use in similarity calculation

        The model training is carried out according to the traditional ReID method
        When using the Extractor class, the input is a list of images,
        and the output is the corresponding features of the images
    """

    def __init__(self, use_cuda=True):
        self.model = VGG(reid=True)
        self.device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
        self.model.to(self.device)

        self.size = (64, 128)
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _preprocess(self, im_crops):
        def _resize(im, size):
            return cv2.resize(im.astype(np.float32) / 255., size)

        im_batch = torch.cat([self.norm(_resize(im, self.size)).unsqueeze(0) for im in im_crops], dim=0).float()
        return im_batch

    def __call__(self, im_crops):
        im_batch = self._preprocess(im_crops)
        with torch.no_grad():
            im_batch = im_batch.to(self.device)
            features = self.model(im_batch)
        return features.cpu().numpy()


if __name__ == '__main__':
    img = torch.rand(8, 3, 128, 64)
    extr = FeatureExtractor()
    feature = extr(img)
    print(feature.shape)
