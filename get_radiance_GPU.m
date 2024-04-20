function radiance = get_radiance_GPU(image, transmission, atmosphere)


 [m, n, ~] = size(image);

 rep_atmosphere = repmat(reshape(atmosphere, [1, 1, 3]), m, n);

 max_transmission = repmat(max(transmission, 0.1), [1, 1, 3]);

 radiance = zeros(1920,2560,3);
 radiance = gpuArray(radiance);
% execute mex CUDA function
  radiance = getRadiance(image,max_transmission,rep_atmosphere,radiance,m,n);

% fetch result
%radiance = fromGPUArray(g_dc, x_width, x_height);
%disp(radiance);

end