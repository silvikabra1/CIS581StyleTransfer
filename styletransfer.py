from __future__ import print_function
import torchvision.transforms.functional as TF
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContentLoss(nn.Module):

    def __init__(self, target, loss_function):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()
        self.loss_function = loss_functions[loss_function]

    def forward(self, input):
        self.loss = self.loss_function(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature, loss_function):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss_function = loss_functions[loss_function]

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = self.loss_function(G, self.target)
        return input


cnn = models.vgg19(pretrained=True).features.to(device).eval()
alexnet = models.alexnet(pretrained=True).features.to(device).eval()


# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


optimizers = {'LBFGS': optim.LBFGS,
              'Adam': optim.Adam,
              'SGD': partial(optim.SGD, lr=1e-4, momentum=0.9)}


def get_input_optimizer(input_img, optimizer):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optimizers[optimizer]([input_img])
    return optimizer


loss_functions = {'mse': F.mse_loss,
                  'l1': F.l1_loss,
                  'smoothl1': F.smooth_l1_loss,
                  'huber': F.huber_loss}


# THIS IS FOR VGG
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_3', 'conv_5', 'conv_7', 'conv_9']

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               loss_function,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    # normalization module
    normalization = Normalization(
        normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    j = 0  # increment every time we see a sequential
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        elif isinstance(layer, nn.Sequential):
            j += 1
            name = 'sequential_{}'.format(j)
        elif isinstance(layer, nn.AdaptiveAvgPool2d):
            name = 'adaptive_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(
                layer.__class__.__name__))
        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target, loss_function)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature, loss_function)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(model, normalization_mean, normalization_std,
                       content_img, style_img, input_img,
                       loss_function='mse', optimizer='LBFGS',
                       num_steps=300,
                       style_weight=80000, content_weight=8):
    """Run the style transfer."""
    # print('Building the style transfer model..')
    # print('model: ', model,
    #        ', loss: ', loss_function,
    #        ' optimizer: ', optimizer)
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                     normalization_mean, normalization_std, style_img, content_img, loss_function)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    output_img = input_img.clone().detach()

    output_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = get_input_optimizer(output_img, optimizer)

    # print('Optimizing..')
    run = [0]
    outer_style_score = 0
    outer_content_score = 0
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                output_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(output_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            nonlocal outer_style_score, outer_content_score
            outer_style_score = style_score
            outer_content_score = content_score
            run[0] += 1
            # if run[0] % 50 == 0:
            #     print("run {}:".format(run))
            #     print('Style Loss : {:4f} Content Loss: {:4f}'.format(
            #         style_score.item(), content_score.item()))
            #     print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        output_img.clamp_(0, 1)

    return output_img, outer_style_score.item(), outer_content_score.item()


# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image):
    # fake batch dimension required to fit network's input dimensions
    #image = TF.resize(torch.Tensor(image), (imsize, imsize))
    return image.to(device, torch.float)


class Styler:
    def __init__(self) -> None:
        print("Styler Initialized")
        self.loss = 'l1'
        self.optimizer = 'LBFGS'
        self.iters = 50
        self.model = 'vgg'

    def preprocess_images(self, content, style):

        content = torch.Tensor(content.numpy()).permute(2, 0, 1)
        content = transforms.Resize(
            (content.size()[1] // 4, content.size()[2] // 4))(content)
        style = torch.Tensor(style).permute(2, 0, 1)
        style = transforms.Resize(
            (content.size()[1], content.size()[2]))(style)

        style = style.unsqueeze(0)
        content = content.unsqueeze(0)
        return image_loader(content), image_loader(style)

    def apply_style(self, content_img, style_img):
        content_img, style_img = self.preprocess_images(content_img, style_img)
        assert content_img.size() == style_img.size()

        input_img = content_img.clone().contiguous()
        output, style_out, content_out = run_style_transfer(self.model, cnn_normalization_mean, cnn_normalization_std,
                                                            content_img, style_img, input_img,
                                                            self.loss, self.optimizer, self.iters)

        return output
