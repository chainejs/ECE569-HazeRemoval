function atmosphere = get_atmosphere_gpu(image, dark_channel)

    [m, n, ~] = size(image);
    
    n_pixels = m * n;
    
    n_search_pixels = floor(n_pixels * 0.01);
    
    dark_vec = reshape(dark_channel, n_pixels, 1);
    
    image_vec = reshape(image, n_pixels, 3);

    [~, indices] = sort(dark_vec, 'descend');
    
    atmosphere = zeros(1,3);
    atmosphere = gpuArray(atmosphere);

    atmosphere = cuGetAtmosphere(m, n, n_pixels, n_search_pixels, dark_vec, image_vec, indices, atmosphere);
    disp(atmosphere);
end