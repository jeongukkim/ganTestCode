import torch
import torch.nn.functional as F
import numpy as np

from torch import nn


class Generator(nn.Module):
  def __init__(self, targetResolution, numFCLayers=8, latentDim=512):
    super(Generator, self).__init__()

    self.mapping = MappingNetwork(latentDim=latentDim, numLayers=numFCLayers)
    self.synthesis = SynthesisNetwork(latentDim=latentDim, targetResolution=targetResolution)

  def forward(self, latent):
    intermediateLatent = self.mapping(latent)
    outputImage = self.synthesis(intermediateLatent)

    return outputImage


class MappingNetwork(nn.Module):
  """ Mapping Network: Z (latent space) -> W (intermediate latent space) """

  def __init__(self, latentDim=512, numLayers=8):
    super(MappingNetwork, self).__init__()

    fullyConnectedBlock = [PixelNorm()]
    for _ in range(numLayers):
      fullyConnectedBlock.append(EqualLinear(latentDim, latentDim))
      fullyConnectedBlock.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))

    self.mapping = nn.Sequential(*fullyConnectedBlock)

  def forward(self, latent):
    return self.mapping(latent)


class PixelNorm(nn.Module):
  """ Normalize Latent Code """

  def __init__(self, eps=1e-8):
    super(PixelNorm, self).__init__()

    self._eps = eps

  def forward(self, latent):
    norm = latent.square().mean(dim=1, keepdim=True).add_(self._eps).rsqrt()
    return latent * norm


class EqualLinear(nn.Module):
  """ Fully Connected Layer """

  def __init__(self, inChannels, outChannels, isBias=True):
    super(EqualLinear, self).__init__()

    initNormalDist = torch.randn(outChannels, inChannels)
    initZeros = torch.zeros(outChannels)

    self.weight = nn.Parameter(initNormalDist)
    self.bias = nn.Parameter(initZeros) if isBias else None

    # Equalized Learning Rate
    # :: times sqrt(2/fan_in)
    self.multiplier = np.sqrt(2. / inChannels)

  def forward(self, inTensor):
    return F.linear(inTensor, weight=self.weight * self.multiplier, bias=self.bias)


class SynthesisNetwork(nn.Module):
  def __init__(self, latentDim, targetResolution):
    super(SynthesisNetwork, self).__init__()

    resolutionToChannels = {
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

    inputResolution = 4
    self.input = ConstantInput(resolutionToChannels[inputResolution], inputResolution)
    self.conv0 = StyleBlock(latentDim, resolutionToChannels[inputResolution], resolutionToChannels[inputResolution], 3)
    self.toRgb0 = ToRgb(latentDim, resolutionToChannels[inputResolution])

    self.convList = nn.ModuleList()
    self.toRgbList = nn.ModuleList()
    for resolution, channel in resolutionToChannels.items():
      if resolution == 4: continue
      self.convList.append(StyleBlock(latentDim, channel, channel, 3, doUpsample=True))
      self.convList.append(StyleBlock(latentDim, channel, channel, 3))
      self.toRgbList.append(ToRgb(latentDim, channel))
      if resolution == targetResolution: break

    self._latentNum = len(self.toRgbList) + len(self.convList) + 2

  def forward(self, latent):
    latent = latent.unsqueeze(1).repeat(1, self._latentNum, 1)

    batchSize = latent.size(0)
    outTensor = self.input(batchSize)
    outTensor = self.conv0(outTensor, latent[:, 0, :])
    imgTensor = self.toRgb0(None, outTensor, latent[:, 1, :])

    i = 2
    for (conv1, conv2, toRgb) in zip(self.convList[::2], self.convList[1::2], self.toRgbList):
      outTensor = conv1(outTensor, latent[:, i, :])
      outTensor = conv2(outTensor, latent[:, i+1, :])
      imgTensor = toRgb(imgTensor, outTensor, latent[:, i+2, :])

      i += 3

    return imgTensor


class ConstantInput(nn.Module):
  def __init__(self, channel=512, size=4):
    super().__init__()

    initNormalDist = torch.randn(1, channel, size, size)
    self.input = nn.Parameter(initNormalDist)

  def forward(self, batchSize):
    x = self.input.repeat(batchSize, 1, 1, 1)

    return x


class StyleBlock(nn.Module):
  """ Network block where ONE style is active """

  def __init__(self, latentDim, inChannels, outChannels, kernelSize, doUpsample=False):
    super(StyleBlock, self).__init__()

    self.conv = ConvLayer(latentDim, inChannels, outChannels, kernelSize, doUpsample=doUpsample, doDemod=True)
    initZeros = torch.zeros(1, outChannels, 1, 1)
    self.bias = nn.Parameter(initZeros)

    self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    self.noise = NoiseInjection()

  def forward(self, inTensor, latent):
    # Fused Convolution
    outTensor = self.conv(inTensor, latent)

    # NoiseInjection -> AddBias -> Activation
    outTensor = self.noise(outTensor)
    outTensor = self.lrelu(outTensor.add_(self.bias))

    return outTensor


class AffineNetwork(nn.Module):
  """ Affine Transformation Network: W (intermediate latent space) -> s (style) """

  def __init__(self, latentDim, styleDim):
    super(AffineNetwork, self).__init__()

    self.affine = EqualLinear(latentDim, styleDim)
    self.affine.bias.data.fill_(1)  # init bias to 1 instead of 0

  def forward(self, latent):
    return self.affine(latent)


class NoiseInjection(nn.Module):
  def __init__(self):
    super().__init__()

    initZeros = torch.zeros([])
    self.weight = nn.Parameter(initZeros)

  def forward(self, inTensor):
    batchSize, _, imgHeight, imgWidth = inTensor.shape
    noise = torch.randn(batchSize, 1, imgHeight, imgWidth)

    return inTensor.add_(self.weight * noise)


class ConvLayer(nn.Module):
  """ Convolution Layer in StyleGAN2 """

  def __init__(self, latentDim, inChannels, outChannels, kernelSize, doUpsample=False, doDownsample=False,
               doDemod=True):
    super(ConvLayer, self).__init__()

    assert doUpsample * doDownsample == 0
    self._doUpsample = doUpsample
    self._doDownsample = doDownsample
    self._doDemod = doDemod

    self.affine = AffineNetwork(latentDim, inChannels)

    h, w = [kernelSize] * 2
    initNormalDist = torch.randn(outChannels, inChannels, h, w)
    self.weight = nn.Parameter(initNormalDist)

    # Equalized Learning Rate
    # :: times sqrt(2/fan_in)
    fanIn = inChannels * h * w
    self.multiplier = np.sqrt(2. / fanIn)

  def forward(self, inTensor, latent):
    # Bilinear filtering if needed
    if self._doUpsample:
      inTensor = F.interpolate(inTensor, None, 2, 'bilinear')
    if self._doDownsample:
      inTensor = F.interpolate(inTensor, None, 0.5, 'bilinear')

    batchSize, inChannels, imgHeight, imgWidth = inTensor.shape

    # Affine transformation
    style = self.affine(latent)

    # Reshape to [batchSize, outChannels, inChannels, kernelSize, kernelSize]
    style = style.reshape(batchSize, 1, -1, 1, 1)
    weight = self.weight.unsqueeze(0)

    # Modulation
    weight = weight * style

    # Demodulation if needed
    if self._doDemod:
      norm = weight.square().sum(dim=[2, 3, 4], keepdim=True).add_(1e-8).rsqrt()
      weight *= norm

    # Reshape and Group Convolution
    inTensor = inTensor.reshape(1, -1, imgHeight, imgWidth)
    weight = weight.reshape(-1, inChannels, *weight.shape[3:])

    outTensor = F.conv2d(inTensor, weight=weight * self.multiplier, padding="same", groups=batchSize)
    outTensor = outTensor.reshape(batchSize, -1, imgHeight, imgWidth)

    return outTensor


class ToRgb(nn.Module):
  """ Convert to RGB using output skips"""

  def __init__(self, latentDim, inChannels):
    super(ToRgb, self).__init__()

    self.toRgb = ConvLayer(latentDim, inChannels, outChannels=3, kernelSize=1, doDemod=False)
    initZeros = torch.zeros(1, 3, 1, 1)
    self.bias = nn.Parameter(initZeros)

  def forward(self, prevTensor, inTensor, latent):
    outTensor = self.toRgb(inTensor, latent).add_(self.bias)

    if prevTensor is not None:
      prevTensor = F.interpolate(prevTensor, None, 2, 'bilinear')
      outTensor += prevTensor

    return outTensor
