from dataclasses import dataclass, field
from typing import List

PP_DOCLAYOUT_L_TO_HOCR = {
    "paragraph_title": "ocr_header",
    "image":           "ocr_float",
    "text":            "ocr_par",
    "page_number":     "ocr_footer",
    "abstract":        "ocr_par",
    "table_of_contents": "ocr_par",
    "figure_title":    "ocr_caption",
    "figure":          "ocr_float",
    "table":           "ocr_float",
    "table_title":     "ocr_caption",
    "figure_caption":  "ocr_caption",
    "formula_number":  "ocr_par",
    "formula":         "ocr_float",
    "references":      "ocr_par",
    "footnotes":       "ocr_footer",
    "header":          "ocr_header",
    "footer":          "ocr_footer",
    "algorithm":       "ocr_float",
    "seal":            "ocr_float",
    "header_image":    "ocr_header",
    "footer_image":    "ocr_footer",
    "aside_text":      "ocr_par",
    "doc_title":       "ocr_header",
}


@dataclass
class Word:
    """Represents a single OCR word with coordinates and OCR text."""
    coordinates: List[float]  # [x_min, y_min, x_max, y_max]
    text: str = ""


@dataclass
class Line:
    """Represents a single OCR line, with its bounding box and nested words."""
    coordinates: List[float] # [x_min, y_min, x_max, y_max]
    words: List[Word] = field(default_factory=list)


@dataclass
class Layout:
    """A Layout can have other layouts nested within, with their own OCR lines."""
    layout_type: str
    coordinates: List[float]  # [x_min, y_min, x_max, y_max]
    children: List["Layout"] = field(default_factory=list)
    ocr_lines: List[Line] = field(default_factory=list)


def overlaps(line_coords: List[float], layout_coords: List[float]) -> bool:
    """
    Checks for overlaps by axis-aligned bounding boxes (AABBs)
    """

    line_x_min, line_y_min, line_x_max, line_y_max = line_coords
    layout_x_min, layout_y_min, layout_x_max, layout_y_max = layout_coords

    if line_x_max <= layout_x_min or layout_x_max <= line_x_min:
        return False  # No horizontal overlap
    if line_y_max <= layout_y_min or layout_y_max <= line_y_min:
        return False  # No vertical overlap

    return True

def contains(line_coords: List[float], layout_coords: List[float]) -> bool:
    """
    Checks if the line is fully contained within the layout.
    """

    line_x_min, line_y_min, line_x_max, line_y_max = line_coords
    layout_x_min, layout_y_min, layout_x_max, layout_y_max = layout_coords

    return (layout_x_min <= line_x_min and
            layout_y_min <= line_y_min and
            layout_x_max >= line_x_max and
            layout_y_max >= line_y_max)
