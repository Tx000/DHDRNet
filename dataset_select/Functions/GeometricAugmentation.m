function output = GeometricAugmentation(input, ind)

if (ind == 1)
    output = input;
end

if (ind == 2)
    output = fliplr(input);
end

if (ind == 3)
    output = flipud(input);
end

if (ind == 4)
    output = rot90(input, 2);
end

if (ind == 5)
    output = rot90(input, 3);
end

if (ind == 6)
    output = rot90(input, 3);
    output = fliplr(output);
end

if (ind == 7)
    output = rot90(input, 3);
    output = flipud(output);
end

if (ind == 8)
    output = rot90(input, 1);
end


