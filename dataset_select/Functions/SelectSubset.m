function inds = SelectSubset(input)

global param;
patchSize = param.patchSize;

maxTH = 0.8;
minTh = 0.2;

thresh = 0.2 * patchSize * patchSize * 3;

badInds = input > maxTH | input < minTh;

inds = sum(sum(sum(badInds, 1), 2), 3) > thresh;
inds = find(inds == 1);