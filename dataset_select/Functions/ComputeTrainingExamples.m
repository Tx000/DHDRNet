function [inputs, label, num] = ComputeTrainingExamples(curImgsLDR, curExpo, curLabel)

global param;

patchSize = param.patchSize;
stride = param.stride;

%%% prepare input features
curInputs = PrepareInputFeatures(curImgsLDR, curExpo);

inputs = GetPatches(curInputs, patchSize, stride);
label = GetPatches(curLabel, patchSize, stride);

selInds = SelectSubset(inputs(:, :, 7:9, :));

inputs = inputs(:, :, :, selInds);
label = label(:, :, :, selInds);
num = length(selInds);


