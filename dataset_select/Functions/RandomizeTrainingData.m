function RandomizeTrainingData()

global param;

trainingFolder = param.trainingData;
outputFolder = param.trainingData;
trainingFile = [trainingFolder, '/', 'TrainSequence.h5'];


chunksz = 1000;
endInd = 0;
count = 0;
%% main loop

info = h5info(trainingFile);
numPatches = info.Datasets(1).Dataspace.Size(end);

rfSize = info.Datasets(1).Dataspace.Size;
inSize = info.Datasets(2).Dataspace.Size;

writeOrder = randperm(numPatches);

for i = 1 : numPatches
    
    curInd = mod(i-1, chunksz) + 1;
    
    if (curInd == 1)
        inputs = zeros([inSize(1:end-1), chunksz], 'single');
        label = zeros([rfSize(1:end-1), chunksz], 'single');
    end
    
    inputs(:, :, :, curInd) = h5read(trainingFile, '/IN', [1, 1, 1, writeOrder(i)], [inSize(1:end-1), 1]);
    label(:, :, :, curInd) = h5read(trainingFile, '/GT', [1, 1, 1, writeOrder(i)], [rfSize(1:end-1), 1]);
    
    if(curInd == chunksz || i == numPatches)
        fprintf(repmat('\b', [1, count]));
        
        inputs = inputs(:, :, :, 1:curInd);
        label = label(:, :, :, 1:curInd);
        
        count = fprintf('Finished randomizing patch %d of %d\n', i, numPatches);
        endInd = WriteTrainingExamples(inputs, label, endInd, [outputFolder, 'Training.h5']);
    end
end

fprintf(repmat('\b', [1, count]));
fprintf('Done\n\n');

delete(trainingFile);