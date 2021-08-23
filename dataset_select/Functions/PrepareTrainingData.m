function PrepareTrainingData()

global param;

sceneFolder = param.trainingScenes;
outputFolder = param.trainingData;

[~, scenePaths, numScenes] = GetFolderContent(sceneFolder, [], true);

endInd = 0;
MakeDir(outputFolder);
num_all = 0;

for i = 1 : numScenes
    
    fprintf('Started working on scene %d of %d \n', i, numScenes);
    
    %%% reading input data
    curExpo = ReadExpoTimes(scenePaths{i});
    [curImgsLDR, curLabel] = ReadImages(scenePaths{i});
    
    %%% processing data
    [inputs, label, num] = ComputeTrainingExamples(curImgsLDR, curExpo, curLabel);
    fprintf('save %d patches \n\n', num);
    num_all = num_all + num;
    
    %%% writing data
    endInd = WriteTrainingExamples(inputs, label, endInd, [outputFolder, 'TrainSequence.h5']);
    
end

fprintf('Totally %d patches\n', num_all);

fprintf('Done\n\n');