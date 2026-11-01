"""Extract graphical approximations from Figure 16a's PDF vector polylines.

Requires pdfplumber. Run from the repository root. These are not raw author data.
The selected polylines and axes were visually verified against page 12.
"""
import json
from pathlib import Path

import pdfplumber

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "reference_signal_area_2023.pdf"
DESTINATION = ROOT / "reference_figure16a_digitized.json"
IDS = {"ED": 20, "CT-SA": 21, "SSA": 22, "DL-ANN Option 1": 75,
       "DL-ANN Option 2": 76, "DL-ANN Option 3": 77, "DL-ANN Option 4": 78}
LEFT, RIGHT, TOP, BOTTOM = 90.9735, 266.0925, 63.0485, 197.0105

with pdfplumber.open(SOURCE) as document:
    page = document.pages[11]
    curves = {}
    for name, index in IDS.items():
        points = page.curves[index]["pts"]
        assert len(points) == 26, (name, "Unexpected PDF curve structure")
        values = []
        for x, y in points:
            snr = -20 + 25 * (x - LEFT) / (RIGHT - LEFT)
            assert abs(snr - round(snr)) < 1e-5
            f1 = (BOTTOM - y) / (BOTTOM - TOP)
            assert 0 <= f1 <= 1
            values.append({"snr_db": round(snr), "f1": f1})
        assert [value["snr_db"] for value in values] == list(range(-20, 6))
        curves[name] = values

result = {
    "source_url": "https://www.lopezbenitez.es/journals/Physical_Communication_2023.pdf",
    "doi": "10.1016/j.phycom.2023.102113",
    "figure": "16a",
    "page_1based": 12,
    "method": "Digitized from vector polyline vertices in public author accepted manuscript; "
              "not author-provided raw measurements. Curves identified visually against "
              "figure legend. Coordinates scaled with verified plot frame. "
              "Values are graphical approximations.",
    "plot_bounds_pdf_points": {"left": LEFT, "right": RIGHT, "top": TOP, "bottom": BOTTOM},
    "curves": curves,
}
DESTINATION.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(DESTINATION)
