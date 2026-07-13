from importlib.metadata import version
from typing import List

import click
import cv2
import numpy as np
import requests
from paddleocr import LayoutDetection, PaddleOCR
from yattag import Doc, indent

from hocr import Layout, Line, Word, overlaps
from utils import scale_image

ocr = PaddleOCR(
  use_doc_orientation_classify=False,
  use_doc_unwarping=False,
  use_textline_orientation=False,
  return_word_box=True)

layout_model = LayoutDetection(model_name="PP-DocLayout-L")


def build_hierarchy(layouts: List[Layout], lines: List[Line]) -> List[Layout]:
  """
  Build a recursive hierarchy of Layout objects with nested OCR lines and words.

  Args:
    layout_data: List of layout boxes from layout_model.predict(), each with 'coordinate' and 'type'
    ocr_data: List of OCR rec_boxes (coordinates only, as [x_min, y_min, x_max, y_max])

  Returns:
    List of root-level Layout objects representing the document hierarchy
  """
  click.secho(f'    Building hierarchy from {len(layouts)} layouts and {len(lines)} OCR lines', fg='cyan')

  # Iterate through the lines and then find the first layout that contains each line, and add the line to that layout's children
  for line in lines:
    for layout in layouts:
      if overlaps(line.coordinates, layout.coordinates):
        layout.children.append(line)
        break  # Stop after the first containing layout is found


def predict_ocr_and_layout(params):
  click.secho(f'    Running OCR Prediction on {params["scaled_img_path"]}', fg='cyan')
  results = ocr.predict(str(params["scaled_img_path"]))
  lines = results[0]
  params['lines'] = []

  for line_box, word_boxes, word_texts in zip(
            lines['rec_boxes'],
            lines['text_word_boxes'],
            lines['text_word']
          ):
    line_coords = [coord / params["scale"] for coord in line_box.tolist()]
    word_coords = [[coord / params["scale"] for coord in word_box] for word_box in word_boxes.tolist()]

    line = Line(coordinates=line_coords, words=[Word(coordinates=wc, text=wt) for wc, wt in zip(word_coords, word_texts)])

    params['lines'].append(line)

  click.secho(f'    Running Layout Prediction on {params["scaled_img_path"]}', fg='cyan')
  results = layout_model.predict(str(params["scaled_img_path"]))
  layouts = results[0]
  params['layouts'] = []

  for box in layouts['boxes']:
    layout_coords = [coord / params["scale"] for coord in box['coordinate']]
    layout = Layout(layout_type=box['label'], coordinates=layout_coords)
    params['layouts'].append(layout)

def visualize_results(params):
  if not params['visualize']:
    return

  # click.secho(f'    Saving visualizations in {params["output_dir"]}', fg='yellow')
    # for res in result:
    #   res.save_to_img(output_dir / f'{page}_ocr.{img_resource.get_format()}')

  # click.secho(f'    Saving visualizations in {params["output_dir"]}', fg='yellow')
    # for layout_res in layout:
    #   layout_res.save_to_img(output_dir / f'{page}_layout.{img_resource.get_format()}')

  # Load the original (unscaled) image
  original_img_path = params["output_dir"] / f'{params["page"]}.{params["img_resource"].get_format()}'
  params['img'] = cv2.imread(str(original_img_path))

  for line in params['lines']:
    x_min, y_min, x_max, y_max = line.coordinates

    # Create 4 corner points for the rectangle
    points = np.array([
      [x_min, y_min],
      [x_max, y_min],
      [x_max, y_max],
      [x_min, y_max]
    ], dtype=np.int32)

    cv2.polylines(params['img'], [points], True, (0, 255, 0), 2)  # Green for OCR

  for layout in params["layouts"]:
    x_min, y_min, x_max, y_max = layout.coordinates

    # Create 4 corner points for the rectangle
    points = np.array([
      [x_min, y_min],
      [x_max, y_min],
      [x_max, y_max],
      [x_min, y_max]
    ], dtype=np.int32)

    cv2.polylines(params['img'], [points], True, (255, 0, 0), 2)  # Blue for layout

  # Save the annotated image
  annotated_path = params['output_dir'] / f"{params['page']}_annotated.{params['img_resource'].get_format()}"
  cv2.imwrite(str(annotated_path), params['img'])
  click.secho(f'    Saved annotated image to {annotated_path}', fg='cyan')


def generate_hocr(params):
  """
  Generate an hOCR file from the OCR and layout results in params.

  Args:
    params: Dictionary containing 'lines', 'layouts', and other relevant data.
  """
  hocr_path = params['output_dir'] / f"{params['page']}.hocr"
  image_path = params['output_dir'] / f"{params['page']}.{params['img_resource'].get_format()}"

  doc, tag, text, line = Doc().ttl()
  doc.asis('<?xml version="1.0" encoding="UTF-8"?>')
  doc.asis('<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">')

  with tag('html', xmlns="http://www.w3.org/1999/xhtml", **{"xml:lang": "en"}, lang="en"):
      with tag('head'):
        line('title', '')
        doc.stag('meta', **{"http-equiv": "Content-Type"}, content="text/html;charset=utf-8")
        doc.stag('meta', name="ocr-system", content=f"paddleocr {version('paddleocr')}")
        doc.stag('meta', name="ocr-capabilities", content="ocr_page ocr_carea ocr_par ocr_line ocrx_word")

  with tag('body'):
    with tag('div', id="page_1", klass = 'ocr_page', title = f'image "{image_path}" bbox 0 0 {params["img_resource"].width} {params["img_resource"].height}'):
      for line in params['lines']:
        with tag('span',
                 klass = 'ocr_line',
                 title = f'bbox {int(line.coordinates[0])} {int(line.coordinates[1])} {int(line.coordinates[2])} {int(line.coordinates[3])}'):
          for word in line.words:
            with tag('span',
                      klass = 'ocrx_word',
                      title = f'bbox {int(word.coordinates[0])} {int(word.coordinates[1])} {int(word.coordinates[2])} {int(word.coordinates[3])}'):
              if word.text == ' ':
                doc.asis('&nbsp;')
              else:
                text(word.text)

  with open(hocr_path, 'w') as f:
    f.write(indent(doc.getvalue()))

  click.secho(f'    Generated hOCR file at {hocr_path}', fg='cyan')


def process(params):
  params['scaled_img_path'] = params['output_dir'] / f"{params['page']}_scaled.{params['img_resource'].get_format()}"

  if not params['scaled_img_path'].exists():
    click.secho(f'    Downloading image to {params["scaled_img_path"]}', fg='yellow')
    img_data = requests.get(params['img_resource'].id).content

    with open(params['output_dir'] / f"{params['page']}.{params['img_resource'].get_format()}", 'wb') as f:
      f.write(img_data)

    img_data, params['scale'] = scale_image(img_data)
    with open(params['scaled_img_path'], 'wb') as f:
      f.write(img_data)

  else:
    longest_side = max(params['img_resource'].width, params['img_resource'].height)
    params['scale'] = 2500 / longest_side

  predict_ocr_and_layout(params)

  # Build hierarchical structure
  # params['hierarchy'] = build_hierarchy(params['layouts'],
                                        # params['lines'])


  visualize_results(params)
  generate_hocr(params)

