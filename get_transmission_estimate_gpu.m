function trans_est = get_transmission_estimate_gpu(image, omega, win_size, atmosphere)
    % Extract input image dimensions
    [m, n, ~] = size(image);
    % Replicate atmosphere values to match the size of the input image
    rep_atmosphere = repmat(reshape(atmosphere, [1, 1, 3]), m, n);
    % Preallocate trans_est on GPU
    trans_est = zeros(m, n, 'gpuArray');
    % Call CUDA function for dark channel estimation
    dark_channel = get_dark_channel_gpu(image ./ rep_atmosphere, win_size);
    % Call CUDA function for transmission estimation
    trans_est = cuGetTransmission(dark_channel, omega, m,n);
    % Compute final transmission estimate using newly found values of dark
    % channel and omega from kernel operations
    trans_est = 1 - omega * dark_channel;
end
