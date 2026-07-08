function [] = plot_binary_image(input_image, marker)

[row,col] = ind2sub(size(input_image), find(input_image == 1));

plot(col, row, marker, 'Color', 'k', 'MarkerFaceColor', 'k')

axis([0 size(input_image,2) 0 size(input_image,1)])