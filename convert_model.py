import sys
import torch
import pickle
from model import Generator


torch.manual_seed(0)


def loadMappingNetwork(srcG, myDict):
    for i in range(8):
        myDict[f"mapping.fcs.{i}.weight"] = srcG.state_dict()[f"mapping.fc{i}.weight"]
        myDict[f"mapping.fcs.{i}.bias"] = srcG.state_dict()[f"mapping.fc{i}.bias"]


def testMappingNetwork(srcG, dstG):
    myDict = {}
    loadMappingNetwork(srcG, myDict)
    # print(myDict.keys())
    dstG.load_state_dict(myDict, strict=False)

    # print(srcG.state_dict()["mapping.fc0.bias"])
    # print(dstG.state_dict()["mapping.mapping.1.bias"])

    latent = torch.randn([1, 512]).cuda()
    c = None

    wSrc = srcG.mapping(latent, c)
    wDst = dstG.mapping(latent)

    wSrc = wSrc.narrow(dim=1, start=0, length=1).squeeze(dim=1).detach().cpu().numpy()
    wDst = wDst.detach().cpu().numpy()

    # print(wSrc)
    # print(wDst)
    print(wSrc - wDst)


def testSynthesisNetwork(srcG, dstG):
    myDict = {}
    loadConstInput(srcG, myDict)
    loadConvBlocks(srcG, myDict)
    loadAffineNetwork(srcG, myDict)
    loadNoiseInjection(srcG, myDict)
    loadToRgb(srcG, myDict)
    # print(myDict.keys())
    dstG.load_state_dict(myDict, strict=False)

    # print(srcG.state_dict()["synthesis.b4.const"])
    # print(dstG.state_dict()["synthesis.input.input"])
    # print(srcG.state_dict()["synthesis.b4.conv1.weight"].shape)
    # print(dstG.state_dict()["synthesis.convList.0.conv.weight"].shape)
    # print(srcG.state_dict()["synthesis.b4.conv1.bias"])
    # print(dstG.state_dict()["synthesis.convList.0.bias"])
    # print(srcG.state_dict()["synthesis.b4.conv1.noise_strength"])
    # print(dstG.state_dict()["synthesis.convList.0.noise.weight"])
    # print(srcG.state_dict()["synthesis.b4.torgb.weight"])
    # print(dstG.state_dict()["synthesis.toRgbList.0.toRgb.weight"])
    # print(srcG.state_dict()["synthesis.b4.torgb.bias"])
    # print(dstG.state_dict()["synthesis.toRgbList.0.bias"])

    latent = torch.randn([1, 512]).cuda()
    c = None

    intermediateLatent = srcG.mapping(latent, c)
    imgSrc = srcG.synthesis(intermediateLatent)
    imgDst = dstG.synthesis(intermediateLatent.narrow(dim=1, start=0, length=1).squeeze(dim=1))
    print(imgDst.abs().max())

    # imgSrc = srcG(latent, c)
    # imgDst = dstG(latent)

    # intermediateLatent = dstG.mapping(latent)
    # imgSrc = srcG.synthesis(intermediateLatent.unsqueeze(0).repeat(1,14,1))
    # imgDst = dstG.synthesis(intermediateLatent)

    print(imgSrc)
    print(imgDst)

    showImg(imgSrc)
    showImg(imgDst)


def loadConstInput(srcG, myDict):
    myDict["synthesis.content"] = srcG.state_dict()["synthesis.b4.const"].unsqueeze(dim=0)


def loadConvBlocks(srcG, myDict):
    myDict["synthesis.4.blocks.0.conv.weight"] = srcG.state_dict()["synthesis.b4.conv1.weight"]
    myDict["synthesis.4.blocks.0.bias"] = srcG.state_dict()["synthesis.b4.conv1.bias"].reshape(1, -1, 1, 1)

    for i in range(3, 9):
        res = int(2 ** i)
        myDict[f"synthesis.{res}.blocks.0.conv.weight"] = srcG.state_dict()[f"synthesis.b{res}.conv0.weight"]
        myDict[f"synthesis.{res}.blocks.0.bias"] = srcG.state_dict()[f"synthesis.b{res}.conv0.bias"].reshape(1, -1, 1, 1)
        myDict[f"synthesis.{res}.blocks.1.conv.weight"] = srcG.state_dict()[f"synthesis.b{res}.conv1.weight"]
        myDict[f"synthesis.{res}.blocks.1.bias"] = srcG.state_dict()[f"synthesis.b{res}.conv1.bias"].reshape(1, -1, 1, 1)


def loadAffineNetwork(srcG, myDict):
    myDict["synthesis.4.blocks.1.conv.affine.weight"] = srcG.state_dict()["synthesis.b4.conv1.affine.weight"]
    myDict["synthesis.4.blocks.1.conv.affine.bias"] = srcG.state_dict()["synthesis.b4.conv1.affine.bias"]

    for i in range(3, 9):
        res = int(2 ** i)
        myDict[f"synthesis.{res}.blocks.0.conv.affine.weight"] = srcG.state_dict()[f"synthesis.b{res}.conv0.affine.weight"]
        myDict[f"synthesis.{res}.blocks.0.conv.affine.bias"] = srcG.state_dict()[f"synthesis.b{res}.conv0.affine.bias"]
        myDict[f"synthesis.{res}.blocks.1.conv.affine.weight"] = srcG.state_dict()[f"synthesis.b{res}.conv1.affine.weight"]
        myDict[f"synthesis.{res}.blocks.1.conv.affine.bias"] = srcG.state_dict()[f"synthesis.b{res}.conv1.affine.bias"]


def loadNoiseInjection(srcG, myDict):
    myDict["synthesis.4.blocks.0.noiseWeight"] = srcG.state_dict()["synthesis.b4.conv1.noise_strength"]

    for i in range(3, 9):
        res = int(2 ** i)
        myDict[f"synthesis.{res}.blocks.0.noiseWeight"] = srcG.state_dict()[f"synthesis.b{res}.conv0.noise_strength"]
        myDict[f"synthesis.{res}.blocks.1.noiseWeight"] = srcG.state_dict()[f"synthesis.b{res}.conv1.noise_strength"]


def loadToRgb(srcG, myDict):
    for i in range(2, 9):
        res = int(2 ** i)
        myDict[f"synthesis.{res}.toRgb.conv.weight"] = srcG.state_dict()[f"synthesis.b{res}.torgb.weight"]
        myDict[f"synthesis.{res}.toRgb.bias"] = srcG.state_dict()[f"synthesis.b{res}.torgb.bias"].reshape(1, -1, 1, 1)
        myDict[f"synthesis.{res}.toRgb.conv.affine.weight"] = srcG.state_dict()[f"synthesis.b{res}.torgb.affine.weight"]
        myDict[f"synthesis.{res}.toRgb.conv.affine.bias"] = srcG.state_dict()[f"synthesis.b{res}.torgb.affine.bias"]


def showImg(genOutput):
    import numpy as np
    import matplotlib.pyplot as plt

    genOutput = genOutput.cpu().detach().numpy().transpose(0, 2, 3, 1)
    genOutput = np.clip((genOutput + 1) / 2, 0, 1)
    genOutput = np.uint8(genOutput * 255).squeeze()

    plt.imshow(genOutput)
    plt.show()


if __name__ == "__main__":
    sys.path.insert(0, "C:\\Projects\\stylegan2-ada-pytorch-main\\")

    with open('C:/Projects/stylegan2-ada-pytorch-main/ckpt/ffhq_256.pkl', 'rb') as f:
        G_ema = pickle.load(f)['G_ema'].cuda()

    G = Generator(targetResolution=256).cuda()

    print(G_ema.state_dict().keys())
    print(G.state_dict().keys())

    testMappingNetwork(G_ema, G)
    testSynthesisNetwork(G_ema, G)



