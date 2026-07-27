from dataclasses import dataclass, field
from typing import List

# https://paddlepaddle.github.io/PaddleX/latest/en/module_usage/tutorials/ocr_modules/layout_detection.html#ii-supported-model-list
# https://kba.github.io/hocr-spec/1.2/#floats-image
HOCR_MAPPINGS = {
  'doc_title': 'ocr_title', # Documentation says 'document title' but it errors?
  'paragraph_title': 'ocr_header',
  'text': 'ocr_par',
  'page_number': 'ocr_pageno',
  'abstract': 'ocr_abstract',
  'table': 'ocr_table',
  'table_title': 'ocr_header',
  'table_of_contents': 'ocr_carea',
  'references': 'ocr_carea',
  'footnotes': 'ocr_carea',
  'header': 'ocr_header',
  'footer': 'ocr_footer',
  'algorithm': 'ocr_float',  # possibly "ocr_math" but it is more restrictive
  'formula': 'ocr_float',
  'formula_number': 'ocr_float',
  'image': 'ocr_image',
  'figure_title': 'ocr_header',
  'figure_caption': 'ocr_caption',
  'seal': 'ocr_float',
  'chart title': 'ocr_header',
  'chart': 'ocr_float',
  'header_image': 'ocr_image',
  'footer_image': 'ocr_image',
  'sidebar text': 'ocr_carea',
  'figure_table title': 'ocr_header',
  'lists of references': 'ocr_carea',
  'number': 'ocr_float'
}


@dataclass
class Word:
  """Represents a single OCR word with coordinates and OCR text."""

  coordinates: List[float]  # [x_min, y_min, x_max, y_max]
  text: str = ''


@dataclass
class Line:
  """Represents a single OCR line, with its bounding box and nested words."""

  coordinates: List[float]  # [x_min, y_min, x_max, y_max]
  words: List[Word] = field(default_factory=list)


@dataclass
class Layout:
  """A Layout can have other layouts nested within, with their own OCR lines."""

  layout_type: str
  coordinates: List[float]  # [x_min, y_min, x_max, y_max]
  children: List['Layout'] = field(default_factory=list)
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

  return (
    layout_x_min <= line_x_min
    and layout_y_min <= line_y_min
    and layout_x_max >= line_x_max
    and layout_y_max >= line_y_max
  )
