function output = CropImg(input, pad1, pad2)

output = input(pad1+1:end-pad1, pad2+1:end-pad2, :, :, :, :, :, :);