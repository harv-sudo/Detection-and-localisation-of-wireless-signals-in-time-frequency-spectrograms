clear
clc

resolution = 'low';

switch resolution
    case 'low'
        res_factor = 1;
    case 'medium'
        res_factor = 5;
    case 'high'
        res_factor = 10;
end

width = 100 * res_factor;
height = 50 * res_factor;

nof_channels = 3;
min_on = 10 * res_factor;
min_off = 5 * res_factor;
lambda_on = 0.5 / res_factor;
lambda_off = 0.5 / res_factor;
channel_map = [0 1 0];
guard_band_fraction = 0.05;

Pfa = 0.01;
SNR_dB_values = -20:1:5;
N = 100;

samples_per_snr = 60000;
chunk_size = 5000;

output_dir = fullfile('..', 'baseline_recreation', 'data', 'raw_large');

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

total_samples = 0;

for SNR_dB = SNR_dB_values
    fprintf('Generating SNR = %d dB\n', SNR_dB);

    chunk_id = 0;

    for start_idx = 1:chunk_size:samples_per_snr
        chunk_id = chunk_id + 1;

        end_idx = min(start_idx + chunk_size - 1, samples_per_snr);
        current_chunk_size = end_idx - start_idx + 1;

        noisy_imgs = zeros(current_chunk_size, height, width, 'single');
        clean_imgs = zeros(current_chunk_size, height, width, 'single');
        labels = zeros(current_chunk_size, 20, 4, 'single');
        nof_SAs_all = zeros(current_chunk_size, 1, 'single');
        SNR_dB_all = SNR_dB * ones(current_chunk_size, 1, 'single');

        for i = 1:current_chunk_size
            [clean_img, nof_SAs, boxes] = create_test_image_channelised_modified( ...
                width, height, nof_channels, lambda_on, min_on, lambda_off, ...
                min_off, channel_map, guard_band_fraction);

            noisy_img = add_noise_to_image(clean_img, 'CFAR', [Pfa SNR_dB N]);

            noisy_imgs(i, :, :) = single(noisy_img);
            clean_imgs(i, :, :) = single(clean_img);
            nof_SAs_all(i) = single(nof_SAs);

            if ~isempty(boxes)
                boxes = boxes(1:min(size(boxes, 1), 20), :);
                labels(i, 1:size(boxes, 1), :) = single(boxes);
            end
        end

        filename = fullfile( ...
            output_dir, ...
            sprintf('snr_%+03d_chunk_%03d.mat', SNR_dB, chunk_id) ...
            );

        save(filename, ...
            'noisy_imgs', ...
            'clean_imgs', ...
            'labels', ...
            'SNR_dB_all', ...
            'nof_SAs_all', ...
            'width', ...
            'height', ...
            'Pfa', ...
            'N', ...
            '-v7.3');

        total_samples = total_samples + current_chunk_size;

        fprintf('  Saved chunk %03d with %d samples\n', chunk_id, current_chunk_size);
    end
end

fprintf('Done. Generated %d samples.\n', total_samples);