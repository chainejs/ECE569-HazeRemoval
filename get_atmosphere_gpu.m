function atmosphere = get_atmosphere_gpu(image, dark_channel)

    [m, n, ~] = size(image);
    
    n_pixels = m * n;
    
    n_search_pixels = floor(n_pixels * 0.01);
    
    dark_vec = reshape(dark_channel, n_pixels, 1);
    disp(dark_vec(1));
    
    image_vec = reshape(image, n_pixels, 3);
    disp(image_vec(1));

    [~, indices] = sort(dark_vec, 'descend');

    %atmosphere = cuGetAtmosphere(image, dark_channel, m, n, n_pixels, n_search_pixels, dark_vec, indices, image_vec);

    % Establish thread grid dimensions
    %k.ThreadBlockSize = [64 1];
    %k.GridSize = [8 1];

    atmosphere = cuGetAtmosphere(m, n, n_pixels, n_search_pixels, dark_vec, image_vec, indices);
    
end