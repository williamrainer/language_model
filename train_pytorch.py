

# TODO: architecture_sampling.Dataset._batchify(include_lengths=True) returns len of each sentence.
# Use lengths as parameter?????
def forwardComputeLoss(self, batch, indvLoss):
    _, context = self.models.encoder.forward(batch)
    loss = 0
    # batchSize = number of sentences in the batch
    indvAvgLoss = torch.zeros(batch.batchSize)

    # TODO: sourceLength = max length in source batch.
    for t in range(batch.sourceLength-1):
        genOutputs = self.models.generator.forward(context.select(2, t))

        # LanguageModel is supposed to predict the following word.
        # TODO: isn't this is always true??
        if t != batch.sourceLength-1:
            #TODO: Get source input batch at timestep `t`. If `t` is None, returns the whole sequence.   onmt/data/Batch.lua
            output = batch.getSourceInput(t+1)

        # Same format with and without features.
        if not torch.is_tensor(output):
            output = torch.tensor(output)

        if indvLoss:
            for i in range(batch.batchSize):
                tmpPred = []
                tmpOutput = []
                for j in range(genOutputs.size(0)):
                    tmpPred.append(genOutputs[j][i, :])
                    tmpOutput.append(output[j][i])
                tmpPred = torch.stack(tmpPred)
                tmpOutput = torch.stack(tmpOutput)

                tmpLoss = self.criterion.forward(tmpPred, tmpOutput)
                indvAvgLoss[i] = indvAvgLoss[i] + tmpLoss
                loss = loss + tmpLoss

        else:
            loss = loss + self.criterion.forward(genOutputs, output)

    if indvLoss:
        # TODO: batch.sourceSize = lengths of each source   (dim = batch x 1)
        indvAvgLoss = torch.div(indvAvgLoss, batch.sourceSize.double())
    return loss, indvAvgLoss



def trainNetwork(self, batch):
    loss = 0
    _, context = self.models.encoder.forward(batch)
    #TODO: zero_grad() ??
    gradContexts = context.clone().zero()

    # For each word of the sentence, generate target.
    for t in range(batch.sourceLength-1):
        genOutputs = self.models.generator.forward(context.select(2, t))

        # LanguageModel is supposed to predict following word.
        if t != batch.sourceLength-1:
            # TODO: Get source input batch at timestep `t`. If `t` is None, returns the whole sequence.   see: onmt/data/Batch.lua
            output = batch.getSourceInput(t + 1)

        # Same format with and without features.
        if not torch.is_tensor(output):
            output = torch.tensor(output)
        loss = loss + self.criterion.forward(genOutputs, output)

        genGradOutput = self.criterion.backward(genOutputs, output)
        for j in range(genGradOutput.size(0)):
            # TODO: batch.size(0)?? see: onmt/data/Batch.lua
            genGradOutput[j] /= batch.batchSize

        #TODO: set it equal to the copy??
        gradContexts[:, t].copy(self.models.generator.backward(context.select(2, t), genGradOutput))

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

