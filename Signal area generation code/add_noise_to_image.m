function noisy_img = add_noise_to_image(clean_img, method, params)

switch method
    
    case 'gaussian'
        
        error('Do not use this one')
        
    case 'specific probabilities'

        error('Do not use this one')

    case 'CFAR'
        
        %   params(1) represents the desired probability of false alarm for CFAR (typically 0.01 to 0.10).
        %   params(2) represents the SNR in dB (this will determine the corresponding probability of detection).
        %   params(3) represents the sensing sample size (N).
        
        Pfa = params(1);
        SNR_dB = params(2);
        N = params(3);
        gamma = 10^(SNR_dB/10);
        
        qfunc_custom = @(x) 0.5 * erfc(x / sqrt(2));
qfuncinv_custom = @(p) sqrt(2) * erfcinv(2*p);

Pd = qfunc_custom((qfuncinv_custom(Pfa)*sqrt(N) - N*gamma)/(sqrt(N)*(1+gamma)));
        Pmd = 1-Pd;                                                         % The miss detection probability

        noisy_img = -1*ones(size(clean_img));
        % Change idle pixels (0) to busy (1) with probability Pfa
        idle_pixels_index = find(clean_img == 0);
        noisy_img(idle_pixels_index) = clean_img(idle_pixels_index) + 1*(rand(size(idle_pixels_index)) < Pfa);
        % Change busy pixels (1) to idle (0) with probability Pmd
        busy_pixels_index = find(clean_img == 1);
        noisy_img(busy_pixels_index) = clean_img(busy_pixels_index) - 1*(rand(size(busy_pixels_index)) < Pmd);
        
    otherwise
        
        error('Unexpected method of noise')
        
end
