import torch
import torch.nn.functional as F

from torch import nn


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
    self.multiplier = torch.sqrt(2. / inChannels)

  def forward(self, inTensor):
    return F.linear(inTensor, weight=self.weight * self.multiplier, bias=self.bias)


class SynthesisNetwork(nn.Module):
  def __init__(self):
    super(SynthesisNetwork, self).__init__()

    pass

class StyleBlock(nn.Module):
  def __init__(self, inChannels, outChannels, kernelSize, isBias=True, latentDim=512):
    super(StyleBlock, self).__init__()

    self.affine = AffineNetwork(latentDim, inChannels)

    h, w = [kernelSize] * 2
    initNormalDist = torch.randn(outChannels, inChannels, h, w)
    initZeros = torch.zeros(outChannels)
    self.weight = nn.Parameter(initNormalDist)
    self.bias = nn.Parameter(initZeros) if isBias else None

    self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    self.noise = NoiseInjection()

  def forward(self, inTensor, latent):
    batchSize, inChannels, imgHeight, imgWidth = inTensor.shape

    # affine transformation
    style = self.affine(latent)

    # reshape to [batchSize, outChannels, inChannels, kernelSize, kernelSize]
    style = style.reshape(batchSize, 1, -1, 1, 1)
    weight = self.weight.unsqueeze(0)

    # Modulation
    weight = weight * style

    # Demodulation
    norm = weight.square().sum(dim=[2, 3, 4], keepdim=True).add_(1e-8).rsqrt()
    weight *= norm

    # reshape
    inTensor = inTensor.reshape(1, -1, imgHeight, imgWidth)
    weight = weight.reshape(-1, inChannels, *weight.shape[3:])

    # Convolution
    outTensor = F.conv2d(inTensor, weight=weight, padding="same", groups=batchSize)
    outTensor = outTensor.reshape(-1, inChannels, imgHeight, imgWidth)

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


# class EqualConv2d(nn.Module):
#   """ 2-dimensional Convolution Layer """
#   def __init__(self, inChannels, outChannels, kernelSize, isBias=False):
#     super(EqualConv2d, self).__init__()
#
#     h, w = [kernelSize] * 2
#     initNormalDist = torch.randn(outChannels, inChannels, h, w)
#     initZeros = torch.zeros(outChannels)
#
#     self.weight = nn.Parameter(initNormalDist)
#     self.bias = nn.Parameter(initZeros) if isBias else None
#
#     # Equalized Learning Rate
#     # :: times sqrt(2/fan_in)
#     fanIn = inChannels * h * w
#     self.multiplier = torch.sqrt(2. / fanIn)
#
#   def forward(self, inTensor):
#     return F.conv2d(inTensor, weight=self.weight * self.multiplier, bias=self.bias, padding="same")

