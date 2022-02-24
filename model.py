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
    1024: 16,
}
SIZE_TO_BLUR_KERNEL = {
    1: [1],
    2: [1, 1],
    3: [1, 2, 1],
    4: [1, 3, 3, 1],
    5: [1, 4, 6, 4, 1],
    6: [1, 5, 10, 10, 5, 1],
    7: [1, 6, 15, 20, 15, 6, 1],
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
        latent = latent * latent.square().mean(dim=1, keepdim=True).add_(EPSILON).rsqrt()
        intermediateLatent = self.fcs(latent)

        return intermediateLatent


class LinearLayer(nn.Module):
    """ Fully Connected Layer (+ Leaky ReLU with the negative slope of 0.2)"""

    def __init__(self, inChannels, outChannels, lrMultiplier=1., doActivation=True):
        super(LinearLayer, self).__init__()

        self._doActivation = doActivation
        self._lrMultiplier = lrMultiplier

        initNormalDist = torch.randn((outChannels, inChannels)) / lrMultiplier
        initZeros = torch.zeros((outChannels))
        self.weight = nn.Parameter(initNormalDist)
        self.bias = nn.Parameter(initZeros)

        # Equalized Learning Rate
        # :: times sqrt(2/fan_in) for ReLU-type activation
        self._weightMultiplier = np.sqrt(1. / inChannels)

    def forward(self, latent):
        w = self.weight * self._weightMultiplier * self._lrMultiplier
        b = self.bias.unsqueeze(dim=0) * self._lrMultiplier

        latent = F.linear(latent, weight=w)

        if self._doActivation:
            # to retain expected signal variance, multiply sqrt(2)
            latent = F.leaky_relu(latent.add_(b), negative_slope=0.2) * np.sqrt(2)
        else:
            latent = latent.add_(b)

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
    """ Network block where ONE style is active + applying noise and bias"""

    def __init__(self, latentDim, inChannels, outChannels, kernelSize, doUpsample=False):
        super(SynthesisBlock, self).__init__()

        self.conv = ConvLayer(latentDim, inChannels, outChannels, kernelSize, doUpsample=doUpsample, doDemod=True)
        initZeros = torch.zeros((1, outChannels, 1, 1))
        self.bias = nn.Parameter(initZeros)

        initZero = torch.zeros([])
        self.noiseWeight = nn.Parameter(initZero)

    def forward(self, content, latent):
        # Style Block
        content = self.conv(content, latent)

        # NoiseInjection
        batchSize, _, cHeight, cWidth = content.shape
        noise = torch.randn((batchSize, 1, cHeight, cWidth), device=content.device)
        content = content.add_(self.noiseWeight * noise)

        # AddBias -> Activation
        # to retain expected signal variance, multiply sqrt(2)
        content = F.leaky_relu(content.add_(self.bias), negative_slope=0.2) * np.sqrt(2)

        return content


class ConvLayer(nn.Module):
    """ Convolution Layer in StyleGAN2 """

    def __init__(self, latentDim, inChannels, outChannels, kernelSize, filterSize=4, doUpsample=False, doDownsample=False, doDemod=True):
        super(ConvLayer, self).__init__()

        assert doUpsample * doDownsample == 0
        self._doUpsample = doUpsample
        self._doDownsample = doDownsample
        self._doDemod = doDemod

        if latentDim is not None:
            self.affine = LinearLayer(latentDim, inChannels, doActivation=False)
            self.affine.bias.data.fill_(1)
        else:
            self.affine = None

        kh, kw = [kernelSize] * 2
        initNormalDist = torch.randn((outChannels, inChannels, kh, kw))
        self.weight = nn.Parameter(initNormalDist)

        # padding
        self._padding = kernelSize // 2

        # blur filter
        # NOT parameter, just state
        filter = SIZE_TO_BLUR_KERNEL[filterSize]
        filter = torch.tensor(filter)
        filter = filter.outer(filter)
        filter = filter / filter.sum()
        self.register_buffer("filter", filter)

        # Equalized Learning Rate
        # :: times sqrt(2/fan_in) for ReLU-type activation
        fanIn = inChannels * kh * kw
        self._weightMultiplier = np.sqrt(1. / fanIn)

    @property
    def isModulationOff(self):
        return self.affine is None

    def forward(self, inTensor, latent):
        weight = self._normalizeWeight(latent)
        outTensor = self._convMain(inTensor, weight)

        return outTensor

    def _normalizeWeight(self, latent):
        if self.isModulationOff:
            weight = self._equalizeLR()
        else:
            weight = self._modulate(latent)

        return weight

    def _equalizeLR(self):
        weight = self.weight * self._weightMultiplier

        return weight

    def _modulate(self, latent):
        styleMultiplier = self._weightMultiplier if not self._doDemod else 1

        style = self.affine(latent) * styleMultiplier
        weight = self.weight

        # Reshape to [batchSize, outChannels, inChannels, kernelSize, kernelSize]
        style = style.reshape(-1, 1, weight.size(dim=1), 1, 1)
        weight = weight.unsqueeze(0)

        # Modulation
        weight = weight * style

        # Demodulation
        if self._doDemod:
            norm = weight.square().sum(dim=[2, 3, 4], keepdim=True).add_(1e-8).rsqrt()
            weight = weight * norm

        return weight

    def _convMain(self, inTensor, weight):
        if self._doUpsample:
            convFunc = self._transeposeStridedConvThenBlur
        elif self._doDownsample:
            convFunc = self._downsampleThenConv if weight.shape[-1] == 1 \
                else self._blurThenStridedConv
        elif self.isModulationOff:
            convFunc = self._plainConv
        else:
            convFunc = self._groupConv

        outTensor = convFunc(inTensor, weight)

        return outTensor

    def _transeposeStridedConvThenBlur(self, inTensor, weight):
        batchSize, _, imgHeight, imgWidth = inTensor.shape

        # transepose strided group convolution
        inTensor = inTensor.reshape(1, -1, imgHeight, imgWidth)
        weight = weight.transpose(1, 2)
        weight = weight.reshape(-1, *weight.shape[2:])

        outTensor = F.conv_transpose2d(inTensor, weight=weight, stride=2, padding=(0, 0), groups=batchSize)

        # Blur
        outTensor = F.pad(outTensor, [1, 1, 1, 1])

        numChannels = outTensor.size(dim=1)

        blurFilter = self.filter
        blurFilter *= 4
        blurFilter = blurFilter[None, None, :, :].repeat(numChannels, 1, 1, 1)
        blurFilter = blurFilter.to(outTensor.device)

        outTensor = F.conv2d(outTensor, weight=blurFilter, groups=numChannels)

        # Reshape to [batchSize, outChannels, imgHeight, imgWidth]
        outTensor = outTensor.reshape(batchSize, -1, *outTensor.shape[2:])

        return outTensor

    def _blurThenStridedConv(self, inTensor, weight):
        # Blur
        outTensor = F.pad(inTensor, [2, 2, 2, 2])

        numChannels = outTensor.size(dim=1)

        blurFilter = self.filter
        blurFilter = blurFilter[None, None, :, :].repeat(numChannels, 1, 1, 1)
        blurFilter = blurFilter.to(outTensor.device)

        outTensor = F.conv2d(outTensor, weight=blurFilter, groups=numChannels)

        # strided convolution
        outTensor = F.conv2d(outTensor, weight=weight, stride=2)

        return outTensor

    def _downsampleThenConv(self, inTensor, weight):
        # downsample
        outTensor = F.pad(inTensor, [1, 1, 1, 1])

        numChannels = outTensor.size(dim=1)

        blurFilter = self.filter
        blurFilter = blurFilter[None, None, :, :].repeat(numChannels, 1, 1, 1)
        blurFilter = blurFilter.to(outTensor.device)

        outTensor = F.conv2d(outTensor, weight=blurFilter, groups=numChannels)
        outTensor = outTensor[:, :, ::2, ::2]

        # convolution
        outTensor = F.conv2d(outTensor, weight=weight)

        return outTensor

    def _plainConv(self, inTensor, weight):
        outTensor = F.conv2d(inTensor, weight=weight, padding=self._padding)

        return outTensor

    def _groupConv(self, inTensor, weight):
        batchSize, _, imgHeight, imgWidth = inTensor.shape

        # Group Convolution sees one sample with N groups
        inTensor = inTensor.reshape(1, -1, imgHeight, imgWidth)
        weight = weight.reshape(-1, *weight.shape[2:])

        outTensor = F.conv2d(inTensor, weight=weight, padding=self._padding, groups=batchSize)

        # Reshape to [batchSize, outChannels, imgHeight, imgWidth]
        outTensor = outTensor.reshape(batchSize, -1, *outTensor.shape[2:])

        return outTensor


class ToRgb(nn.Module):
    """ Convert to RGB using output skips"""

    def __init__(self, latentDim, inChannels, filterSize=4):
        super(ToRgb, self).__init__()

        self.conv = ConvLayer(latentDim, inChannels, outChannels=3, kernelSize=1, doDemod=False)
        initZeros = torch.zeros((1, 3, 1, 1))
        self.bias = nn.Parameter(initZeros)

        # blur filter
        # NOT parameter, just state
        filter = SIZE_TO_BLUR_KERNEL[filterSize]
        filter = torch.tensor(filter)
        filter = filter.outer(filter)
        filter = filter / filter.sum()
        self.register_buffer("filter", filter)

    def forward(self, prevTensor, inTensor, latent):
        outTensor = self.conv(inTensor, latent).add_(self.bias)
        outTensor = self._skipConnectionAfterUpsample(prevTensor, outTensor)

        return outTensor

    def _skipConnectionAfterUpsample(self, prevImg, outputImg):
        if prevImg is not None:
            # Upsampling
            batchSize, inChannel, imageHeight, imageWidth = prevImg.shape
            prevImg = prevImg.reshape(batchSize, inChannel, imageHeight, 1, imageWidth, 1)
            prevImg = F.pad(prevImg, [0, 1, 0, 0, 0, 1])
            prevImg = prevImg.reshape(batchSize, inChannel, imageHeight * 2, imageWidth * 2)
            prevImg = F.pad(prevImg, [2, 1, 2, 1])

            blurFilter = self.filter
            blurFilter *= 4
            blurFilter = blurFilter[None, None, :, :].repeat(inChannel, 1, 1, 1)
            blurFilter = blurFilter.to(outputImg.device)

            prevImg = F.conv2d(prevImg, weight=blurFilter, groups=inChannel)

            # Summing
            outputImg += prevImg

        return outputImg


class Discriminator(nn.Module):
    def __init__(self, targetResolution, minibatchGroup=1):
        super(Discriminator, self).__init__()

        self._minibatchGroup = minibatchGroup

        # list ranging from targetResolution to 8
        self._resolutionGroup = list(reversed([x for x in RES_TO_CHANNELS.keys() if x <= targetResolution]))[:-1]

        inChannel = RES_TO_CHANNELS[targetResolution]
        initZeros = torch.zeros((1, inChannel, 1, 1))
        self.frgbBias = nn.Parameter(initZeros)
        self.fromRgb = ConvLayer(None, 3, inChannel, 1)

        self.downsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for resolution in self._resolutionGroup:
            outChannel = RES_TO_CHANNELS[resolution // 2]
            self.downsamples.append(ConvLayer(None, inChannel, outChannel, kernelSize=1, doDownsample=True))
            self.blocks.append(DiscriminatorBlock(inChannel, outChannel, kernelSize=3))
            inChannel = outChannel

        outChannel = RES_TO_CHANNELS[4]
        self.finalConv = ConvLayer(None, inChannel + self._minibatchGroup, outChannel, kernelSize=3)
        initZeros = torch.zeros((1, outChannel, 1, 1))
        self.finalConvBias = nn.Parameter(initZeros)

        self.finalLinear = nn.Sequential(
            LinearLayer(outChannel * 4 * 4, outChannel),
            LinearLayer(outChannel, 1, doActivation=False)
        )

    def forward(self, image):
        # from Rgb
        inTensor = self.fromRgb(image)
        inTensor = F.leaky_relu(inTensor.add_(self.frgbBias), negative_slope=0.2) * np.sqrt(2)

        # residual
        for downsample, block in zip(self.downsamples, self.blocks):
            downTensor = downsample(inTensor, None)
            outTensor = block(inTensor)

            # adding
            inTensor = (downTensor + outTensor) / np.sqrt(2)

        # minibatch std
        outTensor = self._minibatchStd(inTensor)

        # final conv -> final fully connected
        outTensor = self.finalConv(outTensor)
        outTensor = F.leaky_relu(outTensor.add_(self.finalConvBias), negative_slope=0.2) * np.sqrt(2)

        out = self.finalLinear(outTensor.flatten(start_dim=1))

        return out

    def _minibatchStd(self, inTensor):
        batchSize, inChannel, imageHeight, imageWidth = inTensor.shape

        stddev = inTensor.view(self._minibatchGroup, -1, 1, inChannel, imageHeight, imageWidth)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + EPSILON)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(self._minibatchGroup, 1, imageHeight, imageWidth)

        outTensor = torch.cat([inTensor, stddev], dim=1)

        return outTensor


class DiscriminatorBlock(nn.Module):
    """ Network block where two consecutive conv layers applied """

    def __init__(self, inChannels, outChannels, kernelSize):
        super(DiscriminatorBlock, self).__init__()

        self.conv0 = ConvLayer(None, inChannels, inChannels, kernelSize)
        initZeros = torch.zeros((1, inChannels, 1, 1))
        self.bias0 = nn.Parameter(initZeros)

        self.conv1 = ConvLayer(None, inChannels, outChannels, kernelSize, doDownsample=True)
        initZeros = torch.zeros((1, outChannels, 1, 1))
        self.bias1 = nn.Parameter(initZeros)

    def forward(self, inTensor):
        # conv0 layer
        outTensor = self.conv0(inTensor, None)
        outTensor = F.leaky_relu(outTensor.add_(self.bias0), negative_slope=0.2) * np.sqrt(2)

        # conv1 layer
        outTensor = self.conv1(outTensor, None)
        outTensor = F.leaky_relu(outTensor.add_(self.bias1), negative_slope=0.2) * np.sqrt(2)

        return outTensor
