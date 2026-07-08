function [I, nof_SAs, boxes] = create_test_image_channelised_modified(width, height, nof_channels, lambda_on, min_on, lambda_off, min_off, channel_map, guard_band_fraction)

% This function uses different values!!
% Zero = indicates idle state
% One  = indicates busy state

% Create grayscale image
I = zeros(height, width);

% Count the number of SAs
nof_SAs = 0;
boxes = [];

% For each channel
for i = 1:nof_channels
    
    % Add activity for this channel only if specified in the channel map.
    % If we do nothing when channel_map is zero, then the idle states of
    % the original blank image will remain (i.e., the channel is idle).
    if(channel_map(i) == 1)
        
        % Calculate the edges of this channel (taking into account the fraction
        % dedicated to the guard band, which is split equally between both edges)
        width_per_channel = floor(width/nof_channels);
        left_edge = (i-1)*width_per_channel+1 + round((width/nof_channels)*guard_band_fraction/2);
        right_edge = i*width_per_channel - round((width/nof_channels)*guard_band_fraction/2);
        % fprintf('\nLeft | Right = %d | %d\n', left_edge, right_edge);
        
        % Initialise the counter of the number of vertical/time domain
        % pixels added to this channel
        added_pixels = 0;
        
        % Decide if the first state is busy or idle
        current_state = 1*(rand < 0.5);
        
        % Add on/off periods until the channel is full
        while added_pixels < height
            
            % Generate a random 'duration' for this period
            switch current_state
                case 0 % Add an idle period
                    period_duration = floor(min_off + exprnd(1/lambda_off));
                    % The initial blank image already has 'idles' so we
                    % don't need to change I here, just increase the
                    % counter of the number of added pixels
                case 1 % Add a busy period
                    period_duration = floor(min_on + exprnd(1/lambda_on));
                    % Check that we do not exceed the image height
                    if added_pixels + period_duration > height
                        period_duration = height - added_pixels; % Just add the remaining free pixels
                    end
                    % Replace the 'idle' pixels of the initial blank image with 'busy' pixels
                    I(added_pixels+1:added_pixels+period_duration, left_edge:right_edge) = 1;
                    x = left_edge;
                    y = added_pixels + 1;
                    w = right_edge - left_edge + 1;
                    h = period_duration;

boxes = [boxes; x y w h];
                    % Increase the counter of number of SAs
                    nof_SAs = nof_SAs + 1;
            end
            
            % Increase the number of added pixels
            added_pixels = added_pixels + period_duration;
            
            % Switch state
            current_state = 1 - current_state;

        end
        
    end
    
end