from dataclasses import dataclass, field
from typing import List

from numpy import int64

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
    coordinates: List[int]  # [x_min, y_min, x_max, y_max]
    text: str = ""


@dataclass
class Line:
    """Represents a single OCR line, with its bounding box and nested words."""
    coordinates: List[int] # [x_min, y_min, x_max, y_max]
    words: List[Word] = field(default_factory=list)


@dataclass
class Layout:
    """A Layout can have other layouts nested within, with their own OCR lines."""
    layout_type: str
    coordinates: List[int]  # [x_min, y_min, x_max, y_max]
    children: List["Layout"] = field(default_factory=list)
    ocr_lines: List[Line] = field(default_factory=list)


def overlaps(child_coords: List[int], parent_coords: List[int], overlap_threshold: float = 0.5) -> bool:
    """
    Check if a child bounding box has sufficient overlap with a parent bounding box.

    A box is considered contained if the intersection area between child and parent
    is at least `overlap_threshold` fraction of the child's area.

    Args:
        child_coords: [x_min, y_min, x_max, y_max] for the child box
        parent_coords: [x_min, y_min, x_max, y_max] for the parent box
        overlap_threshold: Minimum fraction of child area that must overlap (0.0-1.0, default 0.5 = 50%)

    Returns:
        True if child has sufficient overlap with parent, False otherwise
    """

    child_x_min, child_y_min, child_x_max, child_y_max = map(int64, child_coords)
    parent_x_min, parent_y_min, parent_x_max, parent_y_max = map(int64, parent_coords)

    inter_x_min = max(child_x_min, parent_x_min)
    inter_y_min = max(child_y_min, parent_y_min)
    inter_x_max = min(child_x_max, parent_x_max)
    inter_y_max = min(child_y_max, parent_y_max)

    if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
        return False

    intersection_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    child_area = (child_x_max - child_x_min) * (child_y_max - child_y_min)

    if child_area <= 0:
        return False

    return intersection_area >= (overlap_threshold * child_area)
