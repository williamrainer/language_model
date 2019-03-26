def forwardComputeLoss(self, batch, indvLoss):
    # vector of sentence lengths
    lengths = batch[0][1]
    _, context = self.models.encoder(batch)
    loss = 0
    indvAvgLoss = torch.zeros(batch.size(0))

    for t in range(max(lengths)-1):
        genOutputs = self.models.generator(context.select(1, t))

        # LanguageModel is supposed to predict the following word.
        if t != max(lengths)-1:
            # Source input batch at timestep `t+1`
            output = batch[:, t+1]

        # Same format with and without features.
        if not torch.is_tensor(output):
            output = torch.tensor(output)

        if indvLoss:
            for i in range(batch.size(0)):
                tmpPred = []
                tmpOutput = []
                for j in range(genOutputs.size(0)):
                    tmpPred.append(genOutputs[j][i, :])
                    tmpOutput.append(output[j][i])
                tmpPred = torch.stack(tmpPred)
                tmpOutput = torch.stack(tmpOutput)

                tmpLoss = self.criterion(tmpPred, tmpOutput)
                indvAvgLoss[i] = indvAvgLoss[i] + tmpLoss
                loss = loss + tmpLoss

        else:
            loss = loss + self.criterion(genOutputs, output)

    if indvLoss:
        indvAvgLoss = torch.div(indvAvgLoss, torch.FloatTensor(lengths))
    return loss, indvAvgLoss



def trainNetwork(self, batch):
    # vector of sentence lengths
    lengths = batch[0][1]
    loss = 0
    _, context = self.models.encoder(batch)
    gradContexts = torch.zeros(batch.size())

    # For each word of the sentence, generate target.
    for t in range(max(lengths)-1):
        genOutputs = self.models.generator(context.select(1, t))

        # LanguageModel is supposed to predict following word.
        if t != max(lengths)-1:
            # Source input batch at timestep `t+1`.
            output = batch[:, t+1]

        # Same format with and without features.
        if not torch.is_tensor(output):
            output = torch.tensor(output)

        loss = loss + self.criterion(genOutputs, output)
        genGradOutput = self.criterion.backward(genOutputs, output)
        for j in range(genGradOutput.size(0)):
            genGradOutput[j] /= batch.size(0)

        gradContexts[:, t] = self.models.generator.backward(context.select(1, t), genGradOutput).copy()

    self.models.encoder.backward(batch, None, gradContexts)
    return loss




# See onmt/modules/ParallelClassNLLCriterion.lua
class ParallelClassNLLCriterion():
    # TODO: inherits from another class?
    parent.__init(self, false)

    def __init__(self, outputSizes):
        for i in range(outputSizes.size()):
            nll = self._addCriterion(outputSizes[i])
            if i == 0:
                self.mainCriterion = nll

    def _addCriterion(self, size):
        # Ignores padding value.
        w = torch.ones(size)
        w[0] = 0

        nll = nn.ClassNLLCriterion(w)

        # Let the training code manage loss normalization.
        nll.sizeAverage = False
        self.add(nll)
        return nll
