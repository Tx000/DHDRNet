function [ imgs_ref ] = PrepareRefFeatures( imgs_ref, curExpo )

curimgs = imgs_ref;

imgs_ref = curimgs{1};
imgs_ref = cat(3, imgs_ref, LDRtoHDR(curimgs{1}, curExpo(1)));
imgs_ref = cat(3, imgs_ref, curimgs{2});
imgs_ref = cat(3, imgs_ref, LDRtoHDR(curimgs{2}, curExpo(3)));

end

