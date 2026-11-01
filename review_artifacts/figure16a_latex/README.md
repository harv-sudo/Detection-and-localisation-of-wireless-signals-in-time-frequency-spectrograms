# Figure 16a overlay in LaTeX

This is a new LaTeX/PGFPlots recreation. The original supplied PDF was made with Python/Matplotlib, using `../plot_original_figure_overlay.py`.

Upload this entire folder (or the supplied ZIP) to Overleaf, choose `main.tex` as the main document and compile with pdfLaTeX. All required assets are included; no Python is needed to compile this project.

- `main.tex`: editable layout, colours, line styles, labels and plots.
- `data/scores.dat`: saved per-SNR scores for the four local models.
- `images/figure16a_original.png`: an unmodified crop of Figure 16a from page 12 of the public author manuscript.

The overlay curves and zoom are drawn by PGFPlots; this does not simply embed the finished Python figure. The paper background is an image. The layout is a close recreation rather than a pixel-identical copy of the Matplotlib output.

The source paper is Alammar and Lopez-Benitez, Signal area estimation based on deep learning (2023), DOI 10.1016/j.phycom.2023.102113. Public manuscript: https://www.lopezbenitez.es/journals/Physical_Communication_2023.pdf .

For a pixel-identical inclusion of the existing PDF in another LaTeX document, use `\usepackage{graphicx}` and `\includegraphics[width=\linewidth]{figure16a_with_our_models.pdf}` after placing that PDF beside your TeX file.
