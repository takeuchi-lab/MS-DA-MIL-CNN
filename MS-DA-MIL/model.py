import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from Adaptive_Grad_Reverse_Layer import AdaptiveGradReverse


class feature_extractor(nn.Module):
    def __init__(self):
        super(feature_extractor, self).__init__()
        vgg16 = models.vgg16(pretrained=False)
        vgg16.load_state_dict(torch.load('vgg16.pth'))
        self.feature_ex = nn.Sequential(*list(vgg16.children())[:-1])
    def forward(self, input):
        x = input.squeeze(0)
        feature = self.feature_ex(x)
        feature = feature.view(feature.size(0), -1)
        return feature


class class_predictor(nn.Module):
    def __init__(self):
        super(class_predictor, self).__init__()
        # reduce feature dimension
        self.feature_extractor_2 = nn.Sequential(
            nn.Linear(in_features=25088, out_features=2048),
            nn.ReLU(),
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU()
        )
        # attention network
        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        # class predictor
        self.classifier = nn.Sequential(
            nn.Linear(512, 2),
        )
    def forward(self, input):
        x = input.squeeze(0)
        H = self.feature_extractor_2(x)
        A = self.attention(H)
        A = torch.transpose(A, 1, 0)
        A = F.softmax(A, dim=1)
        M = torch.mm(A, H)  # KxL
        class_prob = self.classifier(M)
        class_softmax = F.softmax(class_prob, dim=1)
        class_hat = int(torch.argmax(class_softmax, 1))
        return class_prob, class_hat, A


class domain_predictor(nn.Module):
    def __init__(self, domain_num):
        super(domain_predictor, self).__init__()
        # domain predictor
        self.domain_classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, domain_num)
        )
    def forward(self, input):
        x = input.squeeze(0)
        domain_prob = self.domain_classifier(x)
        return domain_prob


class DAMIL(nn.Module):
    def __init__(self, feature_ex, class_predictor, domain_predictor):
        super(DAMIL, self).__init__()
        self.feature_extractor = feature_ex
        self.class_predictor = class_predictor
        self.domain_predictor = domain_predictor

    def forward(self, input, mode, DArate):
        x = input.squeeze(0)
        # extract feature vectors
        features = self.feature_extractor(x)
        # class prediction
        class_prob, class_hat, A = self.class_predictor(features)
        # adaptive DANN in training (mode='train')
        if(mode == 'train'):
            # input to gradient reversal layer
            adapGR_features = AdaptiveGradReverse.apply(features, DArate, A)
            # domain predictor
            domain_prob = self.domain_predictor(adapGR_features)
            return class_prob, domain_prob, class_hat
        # in testing (mode='test')
        if(mode == 'test'):
            return class_prob, class_hat, A


# in the case of multi-scale (x10 and x20)
class MSDAMIL(nn.Module):
    def __init__(self, feature_ex_x10, feature_ex_x20, class_predictor):
        super(MSDAMIL, self).__init__()
        self.feature_extractor_x10 = feature_ex_x10
        self.feature_extractor_x20 = feature_ex_x20
        self.class_predictor = class_predictor
        # fix parameters in feature extractor
        for param in self.feature_extractor_x10.parameters():
            param.requires_grad = False
        for param in self.feature_extractor_x20.parameters():
            param.requires_grad = False

    def forward(self, input_x10, input_x20):
        x10 = input_x10.squeeze(0)
        x20 = input_x20.squeeze(0)
        # extract feature vectors for each magnification
        features_x10 = self.feature_extractor_x10(x10)
        features_x20 = self.feature_extractor_x20(x20)
        # concatnate multi-scale bags
        ms_bag = torch.cat([features_x10, features_x20], dim=0)
        # class prediction
        class_prob, class_hat, A = self.class_predictor(ms_bag)
        return class_prob, class_hat, A
