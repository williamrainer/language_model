function LanguageModel:forwardComputeLoss(batch, indvLoss)
  local _, context = self.models.encoder:forward(batch)

  local loss = 0

  local indvAvgLoss = torch.zeros(batch.size)

  for t = 1, batch.sourceLength-1 do
    local genOutputs = self.models.generator:forward(context:select(2, t))

    -- LanguageModel is supposed to predict the following word.
    local output
    if t ~= batch.sourceLength then
      output = batch:getSourceInput(t + 1)
    end

    -- Same format with and without features.
    if torch.type(output) ~= 'table' then output = { output } end

    if indvLoss then
      for i = 1, batch.size do
        local tmpPred = {}
        local tmpOutput = {}
        for j = 1, #genOutputs do
          table.insert(tmpPred, genOutputs[j][{{i}, {}}])
          table.insert(tmpOutput, output[j][{{i}}])
        end
        local tmpLoss = self.criterion:forward(tmpPred, tmpOutput)
        indvAvgLoss[i] = indvAvgLoss[i] + tmpLoss
        loss = loss + tmpLoss
      end
    else
      loss = loss + self.criterion:forward(genOutputs, output)
    end
  end

  if indvLoss then
    indvAvgLoss = torch.cdiv(indvAvgLoss, batch.sourceSize:double())
  end

  return loss, indvAvgLoss
end

function LanguageModel:trainNetwork(batch)
  local loss = 0

  local _, context = self.models.encoder:forward(batch)

  local gradContexts = context:clone():zero()

  -- For each word of the sentence, generate target.
  for t = 1, batch.sourceLength-1 do
    local genOutputs = self.models.generator:forward(context:select(2, t))

    -- LanguageModel is supposed to predict following word.
    local output
    if t ~= batch.sourceLength then
      output = batch:getSourceInput(t + 1)
    end

    -- Same format with and without features.
    if torch.type(output) ~= 'table' then output = { output } end

    loss = loss + self.criterion:forward(genOutputs, output)

    local genGradOutput = self.criterion:backward(genOutputs, output)
    for j = 1, #genGradOutput do
      genGradOutput[j]:div(batch.totalSize)
    end

    gradContexts[{{}, t}]:copy(self.models.generator:backward(context:select(2, t), genGradOutput))
  end

  self.models.encoder:backward(batch, nil, gradContexts)

  return loss
end
