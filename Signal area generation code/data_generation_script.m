
clear
clc

% All configuration parameters
resolution = 'medium'; % low (50x100) | medium (250x500) | high (500x1000)
switch resolution
    case 'low'
        res_factor = 1;
    case 'medium'
        res_factor = 5;
    case 'high'
        res_factor = 10;
end
width = 100*res_factor;
height = 50*res_factor;
nof_channels = 3;
min_on = 10*res_factor;
min_off = 5*res_factor;
lambda_on = 0.5/res_factor;
lambda_off = 0.5/res_factor;
channel_map = [0 1 0]; % Only central channel is used
guard_band_fraction = 0.05;
Pfa = 0.01;
SNR_dB_values = 5:-1:-15;
N = 100;

plot_images = 1;
FontSize = 16;

for SNR_dB = SNR_dB_values

    % Step 1 - Create clean image
    [clean_img, nof_SAs] = create_test_image_channelised_modified(width, height, nof_channels, lambda_on, min_on, lambda_off, min_off, channel_map, guard_band_fraction);
    if plot_images
        subplot(1,2,1)
        plot_binary_image(clean_img, 'x')
        xlabel('Frequency index', 'Interpreter', 'LaTeX')
        ylabel('Time index', 'Interpreter', 'LaTeX')
        title(sprintf('Original binary spectrogram (%d)', nof_SAs), 'Interpreter', 'LaTeX')
        axis([0 width 0 height])
        set(gca, 'Fontsize', FontSize)
    end

    % Step 2 - Add noise to the test image (CFAR)
    noisy_img = add_noise_to_image(clean_img, 'CFAR', [Pfa SNR_dB N]);
    if plot_images
        subplot(1,2,2)
        plot_binary_image(noisy_img, 'x')
        xlabel('Frequency index', 'Interpreter', 'LaTeX')
        ylabel('Time index', 'Interpreter', 'LaTeX')
        title(sprintf('Noisy binary spect. (SNR = %d dB)', SNR_dB), 'Interpreter', 'LaTeX')
        axis([0 width 0 height])
        set(gca, 'Fontsize', FontSize)
    end

    drawnow
    pause
end
