import torch
import torch.nn as nn
import torch.nn.functional as F
import einx

from .TransformerParts import TransformerBlock


class Encoder(nn.Module):
    def __init__(self, inpDim, embDim, numHeads, numBlocks):
        super().__init__()
        self.stem = nn.Linear(inpDim, embDim)
        self.transformerBlocks = nn.ModuleList([TransformerBlock(embDim, numHeads) for _ in range(numBlocks)])
        self.positionalEmbedder = nn.Embedding(220, embDim)


    def forward(self, x, positionalIndices):
        valueEmbedding = self.stem(x)
        positionalEmbedding = self.positionalEmbedder(positionalIndices).expand(*valueEmbedding.shape)
        embedding = valueEmbedding + positionalEmbedding
        for block in self.transformerBlocks:
            embedding = block(embedding)
        return embedding

class Decoder(nn.Module):
    def __init__(self, outDim, embDim, numHeads, numBlocks):
        super().__init__()
        self.transformerBlocks = nn.ModuleList([TransformerBlock(embDim, numHeads) for _ in range(numBlocks)])
        self.positionalEmbedder = nn.Embedding(220, embDim)
        self.maskingEmbedder = nn.Embedding(1, embDim)
        self.output = nn.Linear(embDim, outDim)
    def forward(self, encoderOutput, allPositionalIndices):
        numMaskedTokens = allPositionalIndices.shape[0] - encoderOutput.shape[1]
        maskedEmbedding = self.maskingEmbedder(torch.tensor(0).to(encoderOutput.device)).expand(encoderOutput.shape[0], numMaskedTokens, encoderOutput.shape[2])
        allEmbeddings = torch.concatenate([encoderOutput, maskedEmbedding], dim=1)
        positionalEmbedding = self.positionalEmbedder(allPositionalIndices).expand(*allEmbeddings.shape)
        embedding = allEmbeddings + positionalEmbedding
        for block in self.transformerBlocks:
            embedding = block(embedding)
        output = self.output(embedding)
        return output

class MaskedAutoencoder(nn.Module):
    def __init__(self,
                inpDim,
                embDim,
                encoderNumHeads,
                encoderNumBlocks,
                decoderNumHeads,
                decoderNumBlocks):
        super().__init__()
        self.inpDim = inpDim
        self.embDim = embDim
        self.encoderNumHeads = encoderNumHeads
        self.encoderNumBlocks = encoderNumBlocks
        self.decoderNumHeads = decoderNumHeads
        self.decoderNumBlocks = decoderNumBlocks

        self.encoder = Encoder(inpDim, embDim, encoderNumHeads, encoderNumBlocks)
        self.decoder = Decoder(inpDim, embDim, decoderNumHeads, decoderNumBlocks)

    
    def forward(self, inpBatch, maskingRatio=0.75):
        firstMaskedIdx = int(inpBatch.shape[1] * (1 - maskingRatio))
        scrambledInputTensor, newPositions = self.scrambleBatch(inpBatch)
        scrambledInputTensor = scrambledInputTensor.to(inpBatch.device)
        newPositions = newPositions.to(inpBatch.device)

        invPermutation = torch.argsort(newPositions)
        
        unmaskedInputTensor = scrambledInputTensor[:, :firstMaskedIdx]
        unmaskedPositions = newPositions[:firstMaskedIdx]

        embeddedOutput = self.encoder(unmaskedInputTensor, unmaskedPositions)

        decodedOutput = self.decoder(embeddedOutput, newPositions)

        maskedInputTensor = scrambledInputTensor[:, firstMaskedIdx:]
        maskedDecodedOutput = decodedOutput[:, firstMaskedIdx:]

        stitchedUpImage = torch.concatenate([unmaskedInputTensor, maskedDecodedOutput], dim=1)
        stitchedInputVizualization = torch.concatenate([unmaskedInputTensor, torch.zeros_like(maskedDecodedOutput)], dim=1)
        unscrambledStitchedImageBatch = einx.get_at("b [p] c, i -> b i c", stitchedUpImage, invPermutation)
        unscrambledStitchedInpViz = einx.get_at("b [p] c, i -> b i c", stitchedInputVizualization, invPermutation)

        return maskedInputTensor, maskedDecodedOutput, unscrambledStitchedInpViz, unscrambledStitchedImageBatch

    
    def scrambleBatch(self, batchedInputTensor):
        newPositions = torch.randperm(batchedInputTensor.shape[1])
        scrambledInputTensor = einx.get_at("b [p] c, i -> b i c", batchedInputTensor, newPositions)
        return scrambledInputTensor, newPositions