function numPatches = GetNumPatches(height, width)

global param;

patchSize = param.patchSize;
stride = param.stride;

numPatchesX = floor((width-patchSize)/stride)+1;
numPatchesY = floor((height-patchSize)/stride)+1;
numPatches = numPatchesY * numPatchesX;
