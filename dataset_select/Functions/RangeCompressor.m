function output = RangeCompressor(input, dzdx)

global mu;

if (~exist('dzdx', 'var') || isempty(dzdx)) 
    output = log(1 + mu * input) / log(1 + mu);
else
    output = dzdx .* (mu / (log(mu+1) .* (mu * input + 1)));
end