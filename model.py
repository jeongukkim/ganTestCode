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
    def __init__(self, target_resolution, latent_dim=512, num_fc_layers=8):
        super(Generator, self).__init__()

        self._latent_dim = latent_dim
        self._target_resolution = target_resolution

        self.mapping = MappingNetwork(latent_dim=latent_dim, num_layers=num_fc_layers)
        self.synthesis = SynthesisNetwork(latent_dim=latent_dim, target_resolution=target_resolution)

    @property
    def image_resolution(self):
        return self._target_resolution

    @property
    def latent_dimension(self):
        return self._latent_dim

    def forward(self, latent):
        intermediate_latent = self.mapping(latent)
        output_image = self.synthesis(intermediate_latent)

        return output_image


class MappingNetwork(nn.Module):
    """ Mapping Network: Z (latent space) -> W (intermediate latent space) """

    def __init__(self, latent_dim=512, num_layers=8):
        super(MappingNetwork, self).__init__()

        self._num_layers = num_layers

        fully_connected_blocks = []
        for _ in range(num_layers):
            fully_connected_blocks.append(LinearLayer(latent_dim, latent_dim, lr_multiplier=0.01))

        self.fcs = nn.Sequential(*fully_connected_blocks)

    @property
    def num_fc_layers(self):
        return self._num_layers

    def forward(self, latent):
        latent = latent * latent.square().mean(dim=1, keepdim=True).add_(EPSILON).rsqrt()
        intermediate_latent = self.fcs(latent)

        return intermediate_latent


class LinearLayer(nn.Module):
    """ Fully Connected Layer (+ Leaky ReLU with the negative slope of 0.2)"""

    def __init__(self, in_channels, out_channels, lr_multiplier=1., do_activation=True):
        super(LinearLayer, self).__init__()

        self._do_activation = do_activation
        self._lr_multiplier = lr_multiplier

        init_weight = torch.randn((out_channels, in_channels)) / lr_multiplier
        self.weight = nn.Parameter(init_weight)
        self.bias = nn.Parameter(torch.zeros((out_channels)))

        # Equalized Learning Rate
        # :: times sqrt(2/fan_in) for ReLU-type activation
        self._weight_multiplier = np.sqrt(1. / in_channels)

    def forward(self, latent):
        w = self.weight * self._weight_multiplier * self._lr_multiplier
        b = self.bias.unsqueeze(dim=0) * self._lr_multiplier

        latent = F.linear(latent, weight=w)

        if self._do_activation:
            # to retain expected signal variance, multiply sqrt(2)
            latent = F.leaky_relu(latent.add_(b), negative_slope=0.2) * np.sqrt(2)
        else:
            latent = latent.add_(b)

        return latent


class SynthesisNetwork(nn.Module):
    """ Synthesis Network """
    def __init__(self, latent_dim, target_resolution):
        super(SynthesisNetwork, self).__init__()

        self._resolution_group = [x for x in RES_TO_CHANNELS.keys() if x <= target_resolution]

        input_resolution = self._resolution_group[0]
        in_channels = RES_TO_CHANNELS[input_resolution]
        const_input = torch.randn((1, in_channels, input_resolution, input_resolution))
        self.content = nn.Parameter(const_input)

        self.synthesis_blocks = nn.ModuleList()
        self.output_skips = nn.ModuleList()
        for res in self._resolution_group:
            out_channels = RES_TO_CHANNELS[res]

            if res == input_resolution:
                self.synthesis_blocks.append(SynthesisBlock(latent_dim, out_channels, out_channels, 3, is_first_block=True))
            else:
                self.synthesis_blocks.append(SynthesisBlock(latent_dim, in_channels, out_channels, 3))

            self.output_skips.append(Skip(latent_dim, out_channels))

            in_channels = out_channels

        self._num_latent = len(self._resolution_group) * 2

    def forward(self, latent):
        latent = latent.unsqueeze(1).repeat(1, self._num_latent, 1)

        batch_size = latent.size(0)
        content = self.content.repeat(batch_size, 1, 1, 1)

        image = None
        latent_idx = 0
        for synthesis_block, output_skip in zip(self.synthesis_blocks, self.output_skips):
            content = synthesis_block(content, latent.narrow(dim=1, start=latent_idx, length=2))
            latent_idx += synthesis_block.num_conv

            image = output_skip(image, content, latent[:, latent_idx, :])

        return image


class Skip(nn.Module):
    """ Convert to RGB using output skips"""

    def __init__(self, latent_dim, in_channels, filter_size=4):
        super(Skip, self).__init__()

        self.torgb = ConvLayer(latent_dim, in_channels, out_channels=3, kernel_size=1, do_demod=False)
        self.bias = nn.Parameter(torch.zeros((1, 3, 1, 1)))

        # blur filter
        # NOT parameter, just state
        filter = SIZE_TO_BLUR_KERNEL[filter_size]
        filter = torch.tensor(filter)
        filter = filter.outer(filter)
        filter = filter / filter.sum()
        self.register_buffer("filter", filter)

    def forward(self, previous_image, in_tensor, latent):
        out_image = self.torgb(in_tensor, latent).add_(self.bias)
        out_image = self._upsample_then_skip_connect(previous_image, out_image)

        return out_image

    def _upsample_then_skip_connect(self, previous_image, output_image):
        if previous_image is not None:
            # Upsampling
            batch_size, in_channels, img_height, img_width = previous_image.shape
            previous_image = previous_image.reshape(batch_size, in_channels, img_height, 1, img_width, 1)
            previous_image = F.pad(previous_image, [0, 1, 0, 0, 0, 1])
            previous_image = previous_image.reshape(batch_size, in_channels, img_height * 2, img_width * 2)
            previous_image = F.pad(previous_image, [2, 1, 2, 1])

            blur_filter = self.filter
            blur_filter *= 4
            blur_filter = blur_filter[None, None, :, :].repeat(in_channels, 1, 1, 1)
            blur_filter = blur_filter.to(output_image.device)

            previous_image = F.conv2d(previous_image, weight=blur_filter, groups=in_channels)

            # Summing
            output_image += previous_image

        return output_image


class SynthesisBlock(nn.Module):
    """ two consecutive style blocks"""

    def __init__(self, latent_dim, in_channels, out_channels, kernel_size, is_first_block=False):
        super(SynthesisBlock, self).__init__()

        self._is_first_block = is_first_block
        self._num_conv = 0

        if not is_first_block:
            self.style_block_0 = ConvLayer(latent_dim, in_channels, out_channels, kernel_size, do_upsample=True, do_demod=True)
            self.style_block_0_bias = nn.Parameter(torch.zeros((1, out_channels, 1, 1)))

            self.noise_weight_0 = nn.Parameter(torch.zeros([]))

            self._num_conv += 1

        self.style_block_1 = ConvLayer(latent_dim, out_channels, out_channels, kernel_size, do_demod=True)
        self.style_block_1_bias = nn.Parameter(torch.zeros((1, out_channels, 1, 1)))

        self.noise_weight_1 = nn.Parameter(torch.zeros([]))

        self._num_conv += 1

    @property
    def num_conv(self):
        return self._num_conv

    def forward(self, content, latent):
        target_latent_index = 0
        if not self._is_first_block:
            block_kwargs = {
                "latent": latent[:, target_latent_index, :],
                "style_block": self.style_block_0,
                "style_block_bias": self.style_block_0_bias,
                "noise_weight": self.noise_weight_0,
            }
            content = self._forward_block(content, **block_kwargs)

            target_latent_index += 1

        block_kwargs = {
            "latent": latent[:, target_latent_index, :],
            "style_block": self.style_block_1,
            "style_block_bias": self.style_block_1_bias,
            "noise_weight": self.noise_weight_1,
        }
        content = self._forward_block(content, **block_kwargs)

        return content

    def _forward_block(self, content, latent, style_block, style_block_bias, noise_weight):
        # Style Block
        content = style_block(content, latent)

        # NoiseInjection
        batch_size, _, content_height, content_width = content.shape
        noise = content.new_empty((batch_size, 1, content_height, content_width)).normal_()
        content = content.add_(noise_weight * noise)

        # AddBias -> Activation
        # to retain expected signal variance, multiply sqrt(2)
        content = F.leaky_relu(content.add_(style_block_bias), negative_slope=0.2) * np.sqrt(2)

        return content


class ConvLayer(nn.Module):
    """ Convolution Layer in StyleGAN2 """

    def __init__(self, latent_dim, in_channels, out_channels, kernel_size, filter_size=4, do_upsample=False, do_downsample=False, do_demod=True):
        super(ConvLayer, self).__init__()

        assert do_upsample * do_downsample == 0
        self._do_upsample = do_upsample
        self._do_downsample = do_downsample
        self._do_demod = do_demod

        if latent_dim is not None:
            self.affine = LinearLayer(latent_dim, in_channels, do_activation=False)
            self.affine.bias.data.fill_(1)
        else:
            self.affine = None

        kernel_height, kernel_width = [kernel_size] * 2
        init_weight = torch.randn((out_channels, in_channels, kernel_height, kernel_width))
        self.weight = nn.Parameter(init_weight)

        # padding
        self._padding = kernel_size // 2

        # blur filter
        # NOT parameter, just state
        filter = SIZE_TO_BLUR_KERNEL[filter_size]
        filter = torch.tensor(filter)
        filter = filter.outer(filter)
        filter = filter / filter.sum()
        self.register_buffer("filter", filter)

        # Equalized Learning Rate
        # :: times sqrt(2/fan_in) for ReLU-type activation
        fan_in = in_channels * kernel_height * kernel_width
        self._weight_multiplier = np.sqrt(1. / fan_in)

    @property
    def is_modulation_off(self):
        return self.affine is None

    def forward(self, in_tensor, latent):
        weight = self._normalize_weight(latent)
        out_tensor = self._conv_main(in_tensor, weight)

        return out_tensor

    def _normalize_weight(self, latent):
        if self.is_modulation_off:
            weight = self._equalize_lr()
        else:
            weight = self._modulate(latent)

        return weight

    def _equalize_lr(self):
        weight = self.weight * self._weight_multiplier

        return weight

    def _modulate(self, latent):
        style_multiplier = self._weight_multiplier if not self._do_demod else 1

        style = self.affine(latent) * style_multiplier
        weight = self.weight

        # Reshape to [batch_size, out_channels, in_channels, kernel_size, kernel_size]
        style = style.reshape(-1, 1, weight.size(dim=1), 1, 1)
        weight = weight.unsqueeze(0)

        # Modulation
        weight = weight * style

        # Demodulation
        if self._do_demod:
            norm = weight.square().sum(dim=[2, 3, 4], keepdim=True).add_(1e-8).rsqrt()
            weight = weight * norm

        return weight

    def _conv_main(self, in_tensor, weight):
        if self._do_upsample:
            conv_func = self._transepose_strided_conv_then_blur
        elif self._do_downsample:
            conv_func = self._downsample_then_conv if weight.shape[-1] == 1 \
                else self._blur_then_strided_conv
        elif self.is_modulation_off:
            conv_func = self._plain_conv
        else:
            conv_func = self._group_conv

        out_tensor = conv_func(in_tensor, weight)

        return out_tensor

    def _transepose_strided_conv_then_blur(self, in_tensor, weight):
        batch_size, _, img_height, img_width = in_tensor.shape

        # transepose strided group convolution
        in_tensor = in_tensor.reshape(1, -1, img_height, img_width)
        weight = weight.transpose(1, 2)
        weight = weight.reshape(-1, *weight.shape[2:])

        out_tensor = F.conv_transpose2d(in_tensor, weight=weight, stride=2, padding=(0, 0), groups=batch_size)

        # Blur
        out_tensor = F.pad(out_tensor, [1, 1, 1, 1])

        num_channels = out_tensor.size(dim=1)

        blur_filter = self.filter
        blur_filter *= 4
        blur_filter = blur_filter[None, None, :, :].repeat(num_channels, 1, 1, 1)
        blur_filter = blur_filter.to(out_tensor.device)

        out_tensor = F.conv2d(out_tensor, weight=blur_filter, groups=num_channels)

        # Reshape to [batch_size, out_channels, img_height, img_width]
        out_tensor = out_tensor.reshape(batch_size, -1, *out_tensor.shape[2:])

        return out_tensor

    def _blur_then_strided_conv(self, in_tensor, weight):
        # Blur
        out_tensor = F.pad(in_tensor, [2, 2, 2, 2])

        num_channels = out_tensor.size(dim=1)

        blur_filter = self.filter
        blur_filter = blur_filter[None, None, :, :].repeat(num_channels, 1, 1, 1)
        blur_filter = blur_filter.to(out_tensor.device)

        out_tensor = F.conv2d(out_tensor, weight=blur_filter, groups=num_channels)

        # strided convolution
        out_tensor = F.conv2d(out_tensor, weight=weight, stride=2)

        return out_tensor

    def _downsample_then_conv(self, in_tensor, weight):
        # downsample
        out_tensor = F.pad(in_tensor, [1, 1, 1, 1])

        num_channels = out_tensor.size(dim=1)

        blur_filter = self.filter
        blur_filter = blur_filter[None, None, :, :].repeat(num_channels, 1, 1, 1)
        blur_filter = blur_filter.to(out_tensor.device)

        out_tensor = F.conv2d(out_tensor, weight=blur_filter, groups=num_channels)
        out_tensor = out_tensor[:, :, ::2, ::2]

        # convolution
        out_tensor = F.conv2d(out_tensor, weight=weight)

        return out_tensor

    def _plain_conv(self, in_tensor, weight):
        out_tensor = F.conv2d(in_tensor, weight=weight, padding=self._padding)

        return out_tensor

    def _group_conv(self, in_tensor, weight):
        batch_size, _, img_height, img_width = in_tensor.shape

        # Group Convolution sees one sample with N groups
        in_tensor = in_tensor.reshape(1, -1, img_height, img_width)
        weight = weight.reshape(-1, *weight.shape[2:])

        out_tensor = F.conv2d(in_tensor, weight=weight, padding=self._padding, groups=batch_size)

        # Reshape to [batch_size, out_channels, img_height, img_width]
        out_tensor = out_tensor.reshape(batch_size, -1, *out_tensor.shape[2:])

        return out_tensor


class Discriminator(nn.Module):
    def __init__(self, target_resolution, minibatch_group=1):
        super(Discriminator, self).__init__()

        self._minibatch_group = minibatch_group

        # list ranging from targetResolution to 8
        self._resolution_group = list(reversed([x for x in RES_TO_CHANNELS.keys() if x <= target_resolution]))[:-1]

        in_channels = RES_TO_CHANNELS[target_resolution]
        self.fromrgb = ConvLayer(None, 3, in_channels, 1)
        self.fromrgb_bias = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))

        self.downsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for resolution in self._resolution_group:
            out_channels = RES_TO_CHANNELS[resolution // 2]
            self.downsamples.append(ConvLayer(None, in_channels, out_channels, kernel_size=1, do_downsample=True))
            self.blocks.append(DiscriminatorBlock(in_channels, out_channels, kernel_size=3))
            in_channels = out_channels

        out_channels = RES_TO_CHANNELS[4]
        self.final_conv = ConvLayer(None, in_channels + self._minibatch_group, out_channels, kernel_size=3)
        initZeros = torch.zeros((1, out_channels, 1, 1))
        self.final_conv_bias = nn.Parameter(initZeros)

        self.final_linear = nn.Sequential(
            LinearLayer(out_channels * 4 * 4, out_channels),
            LinearLayer(out_channels, 1, do_activation=False)
        )

    def forward(self, image):
        # from Rgb
        in_tensor = self.fromrgb(image)
        in_tensor = F.leaky_relu(in_tensor.add_(self.fromrgb_bias), negative_slope=0.2) * np.sqrt(2)

        # residual
        for downsample, block in zip(self.downsamples, self.blocks):
            downTensor = downsample(in_tensor, None)
            out_tensor = block(in_tensor)

            # adding
            in_tensor = (downTensor + out_tensor) / np.sqrt(2)

        # minibatch std
        out_tensor = self._minibatch_std(in_tensor)

        # final conv -> final fully connected
        out_tensor = self.final_conv(out_tensor)
        out_tensor = F.leaky_relu(out_tensor.add_(self.final_conv_bias), negative_slope=0.2) * np.sqrt(2)

        out = self.final_linear(out_tensor.flatten(start_dim=1))

        return out

    def _minibatch_std(self, in_tensor):
        batch_size, in_channels, img_height, img_width = in_tensor.shape

        stddev = in_tensor.view(self._minibatch_group, -1, 1, in_channels, img_height, img_width)
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + EPSILON)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(self._minibatch_group, 1, img_height, img_width)

        out_tensor = torch.cat([in_tensor, stddev], dim=1)

        return out_tensor


class DiscriminatorBlock(nn.Module):
    """ Network block where two consecutive conv layers applied """

    def __init__(self, in_channels, out_channels, kernel_size):
        super(DiscriminatorBlock, self).__init__()

        self.conv0 = ConvLayer(None, in_channels, in_channels, kernel_size)
        self.bias0 = nn.Parameter(torch.zeros((1, in_channels, 1, 1)))

        self.conv1 = ConvLayer(None, in_channels, out_channels, kernel_size, do_downsample=True)
        self.bias1 = nn.Parameter(torch.zeros((1, out_channels, 1, 1)))

    def forward(self, in_tensor):
        # conv0 layer
        out_tensor = self.conv0(in_tensor, None)
        out_tensor = F.leaky_relu(out_tensor.add_(self.bias0), negative_slope=0.2) * np.sqrt(2)

        # conv1 layer
        out_tensor = self.conv1(out_tensor, None)
        out_tensor = F.leaky_relu(out_tensor.add_(self.bias1), negative_slope=0.2) * np.sqrt(2)

        return out_tensor
