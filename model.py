import torch
import torch.nn.functional as F
import numpy as np

from torch import nn


EPSILON = 1e-8
RES_TO_CHANNELS = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 256,
    128: 128,
    256: 64,
    512: 32,
    1024: 16
}


class Generator(nn.Module):
    def __init__(self, targetResolution, latentDim=512, numFCLayers=8):
        super(Generator, self).__init__()

        self._latentDim = latentDim
        self._targetResolution = targetResolution

        self.mapping = MappingNetwork(latentDim=latentDim, numLayers=numFCLayers)
        self.synthesis = SynthesisNetwork(latentDim=latentDim, targetResolution=targetResolution)

    @property
    def image_resolution(self):
        return self._targetResolution

    @property
    def latent_dimension(self):
        return self._latentDim

    def forward(self, latent):
        intermediateLatent = self.mapping(latent)
        outputImage = self.synthesis(intermediateLatent)

        return outputImage


class MappingNetwork(nn.Module):
    """ Mapping Network: Z (latent space) -> W (intermediate latent space) """

    def __init__(self, latentDim=512, numLayers=8):
        super(MappingNetwork, self).__init__()

        self._numLayers = numLayers

        fullyConnectedBlocks = []
        for _ in range(numLayers):
            fullyConnectedBlocks.append(LinearLayer(latentDim, latentDim, lrMultiplier=0.01))

        self.fcs = nn.Sequential(*fullyConnectedBlocks)

    @property
    def num_fc_layers(self):
        return self._numLayers

    def forward(self, latent):
        # latent = latent * latent.square().mean(dim=1, keepdim=True).add_(EPSILON).rsqrt()
        latent = latent * (latent.square().mean(dim=1, keepdim=True)+EPSILON).rsqrt()
        intermediateLatent = self.fcs(latent)

        return intermediateLatent


class LinearLayer(nn.Module):
    """ Fully Connected Layer + Leaky ReLU (alpha=0.2)"""

    def __init__(self, inChannels, outChannels, lrMultiplier=1.):
        super(LinearLayer, self).__init__()

        self._lrMultiplier = lrMultiplier

        initNormalDist = torch.randn((outChannels, inChannels)) / lrMultiplier
        initZeros = torch.zeros((outChannels))
        self.weight = nn.Parameter(initNormalDist)
        self.bias = nn.Parameter(initZeros)

        # Equalized Learning Rate
        # :: times sqrt(2/fan_in)
        self._weightMultiplier = np.sqrt(1. / inChannels)

    def forward(self, latent, lrelu=True):
        w = self.weight * self._weightMultiplier * self._lrMultiplier
        b = self.bias.unsqueeze(dim=0) * self._lrMultiplier

        latent = F.linear(latent, weight=w)
        if lrelu:
            latent = F.leaky_relu(latent.add_(b), negative_slope=0.2) * np.sqrt(2)
        else:
            latent.add_(b)

        return latent


class SynthesisNetwork(nn.Module):
    """ Synthesis Network """
    def __init__(self, latentDim, targetResolution):
        super(SynthesisNetwork, self).__init__()

        self._resolutionGroup = [x for x in RES_TO_CHANNELS.keys() if x <= targetResolution]

        inputResolution = self._resolutionGroup[0]
        inChannel = RES_TO_CHANNELS[inputResolution]
        initNormalDist = torch.randn((1, inChannel, inputResolution, inputResolution))
        self.content = nn.Parameter(initNormalDist)

        for res in self._resolutionGroup:
            outChannel = RES_TO_CHANNELS[res]

            styleBlocks = nn.ModuleList()
            if res != inputResolution:
                styleBlocks.append(SynthesisBlock(latentDim, inChannel, outChannel, 3, doUpsample=True))
            styleBlocks.append(SynthesisBlock(latentDim, outChannel, outChannel, 3))
            setattr(self, f'{res}.blocks', styleBlocks)

            toRgbBlock = ToRgb(latentDim, outChannel)
            setattr(self, f'{res}.toRgb', toRgbBlock)

            inChannel = outChannel

        self._numLatent = len(self._resolutionGroup) * 2

    def forward(self, latent):
        latent = latent.unsqueeze(1).repeat(1, self._numLatent, 1)

        batchSize = latent.size(0)
        content = self.content.repeat(batchSize, 1, 1, 1)

        image = None
        latentIdx = 0
        for res in self._resolutionGroup:
            blocks = getattr(self, f'{res}.blocks')
            for block in blocks:
                content = block(content, latent[:, latentIdx, :])
                latentIdx += 1

            rgbBlock = getattr(self, f'{res}.toRgb')
            image = rgbBlock(image, content, latent[:, latentIdx, :])

        return image


class SynthesisBlock(nn.Module):
    """ Network block where ONE style is active """

    def __init__(self, latentDim, inChannels, outChannels, kernelSize, doUpsample=False):
        super(SynthesisBlock, self).__init__()

        self.conv = ConvLayer(latentDim, inChannels, outChannels, kernelSize, doUpsample=doUpsample, doDemod=True)
        initZeros = torch.zeros((1, outChannels, 1, 1))
        self.bias = nn.Parameter(initZeros)

        initZero = torch.zeros([])
        self.noiseWeight = nn.Parameter(initZero)

    def forward(self, content, latent):
        # Fused Convolution
        content = self.conv(content, latent)

        # NoiseInjection
        batchSize, _, cHeight, cWidth = content.shape
        noise = torch.randn((batchSize, 1, cHeight, cWidth), device=content.device)
        content = content.add_(self.noiseWeight * noise)

        # AddBias -> Activation
        content = F.leaky_relu(content.add_(self.bias), negative_slope=0.2) * np.sqrt(2)

        return content


class ConvLayer(nn.Module):
    """ Convolution Layer in StyleGAN2 """

    def __init__(self, latentDim, inChannels, outChannels, kernelSize, doUpsample=False, doDownsample=False, doDemod=True):
        super(ConvLayer, self).__init__()

        assert doUpsample * doDownsample == 0
        self._doUpsample = doUpsample
        self._doDownsample = doDownsample
        self._doDemod = doDemod

        self.affine = LinearLayer(latentDim, inChannels)
        self.affine.bias.data.fill_(1)

        h, w = [kernelSize] * 2
        initNormalDist = torch.randn((outChannels, inChannels, h, w))
        self.weight = nn.Parameter(initNormalDist)

        # padding
        self._padding = kernelSize // 2

        # blur filter
        self._filter = [1, 3, 3, 1]

        # Equalized Learning Rate
        # :: times sqrt(2/fan_in)
        fanIn = inChannels * h * w
        self._weightMultiplier = np.sqrt(1. / fanIn)

    def get_filter(self):
        filter = torch.tensor(self._filter)
        filter = filter.outer(filter)
        filter = filter / filter.sum()

        return filter

    def forward(self, inTensor, latent, styleMultiplier=1.):
        # Bilinear filtering if needed
        # if self._doUpsample:
        #     inTensor = F.interpolate(inTensor, None, 2, 'bilinear')
        # if self._doDownsample:
        #     inTensor = F.interpolate(inTensor, None, 0.5, 'bilinear')

        batchSize, inChannels, imgHeight, imgWidth = inTensor.shape

        # Affine transformation
        style = self.affine(latent, lrelu=False) * styleMultiplier

        # Equalized LR
        # weight = self.weight * self._weightMultiplier
        weight = self.weight

        # Reshape to [batchSize, outChannels, inChannels, kernelSize, kernelSize]
        style = style.reshape(batchSize, 1, -1, 1, 1)
        weight = weight.unsqueeze(0)

        # Modulation
        weight = weight * style

        # Demodulation if needed
        if self._doDemod:
            # norm = weight.square().sum(dim=[2, 3, 4], keepdim=True).add_(1e-8).rsqrt()
            norm = (weight.square().sum(dim=[2, 3, 4], keepdim=True)+1e-8).rsqrt()
            weight = weight * norm

        # Reshape and Group Convolution
        inTensor = inTensor.reshape(1, -1, imgHeight, imgWidth)
        if self._doUpsample:
            weight = weight.transpose(1, 2)
            weight = weight.reshape(-1, *weight.shape[2:])

            outTensor = F.conv_transpose2d(inTensor, weight=weight, stride=2, padding=(0, 0), groups=batchSize)

            outTensor = F.pad(outTensor, [1, 1, 1, 1])

            channels = outTensor.size(dim=1)

            blur_filter = self.get_filter()
            blur_filter *= 4
            blur_filter = blur_filter[None, None, :, :].repeat(channels, 1, 1, 1)
            blur_filter = blur_filter.to(outTensor.device)

            outTensor = F.conv2d(outTensor, weight=blur_filter, groups=channels)

        else:
            weight = weight.reshape(-1, *weight.shape[2:])

            outTensor = F.conv2d(inTensor, weight=weight, padding=self._padding, groups=batchSize)

        outTensor = outTensor.reshape(batchSize, -1, *outTensor.shape[2:])

        return outTensor


class ToRgb(nn.Module):
    """ Convert to RGB using output skips"""

    def __init__(self, latentDim, inChannels):
        super(ToRgb, self).__init__()

        self.conv = ConvLayer(latentDim, inChannels, outChannels=3, kernelSize=1, doDemod=False)
        initZeros = torch.zeros((1, 3, 1, 1))
        self.bias = nn.Parameter(initZeros)

        fanIn = inChannels
        self._multiplier = np.sqrt(1. / fanIn)

    def forward(self, prevTensor, inTensor, latent):
        outTensor = self.conv(inTensor, latent, styleMultiplier=self._multiplier).add_(self.bias)

        if prevTensor is not None:
            batchSize, inChannel, imageHeight, imageWidth = prevTensor.shape
            prevTensor = prevTensor.reshape(batchSize, inChannel, imageHeight, 1, imageWidth, 1)
            prevTensor = F.pad(prevTensor, [0, 1, 0, 0, 0, 1])
            prevTensor = prevTensor.reshape(batchSize, inChannel, imageHeight * 2, imageWidth * 2)
            prevTensor = F.pad(prevTensor, [2, 1, 2, 1])

            filter = torch.tensor([1,3,3,1])
            filter = filter.outer(filter)
            filter = filter / filter.sum()
            filter *= 4
            filter = filter[None, None, :, :].repeat(inChannel, 1, 1, 1)
            filter = filter.to(outTensor.device)

            prevTensor = F.conv2d(prevTensor, weight=filter, groups=inChannel)

            outTensor += prevTensor

        return outTensor
