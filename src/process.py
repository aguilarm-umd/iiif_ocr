import click
import cv2
import numpy as np
import requests
from paddleocr import LayoutDetection, PaddleOCR

from hocr import HierarchyNode, LayoutBox, OCRBox, is_box_contained
from utils import scale_image

ocr = PaddleOCR(
  use_doc_orientation_classify=False,
  use_doc_unwarping=False,
  use_textline_orientation=False)

layout_model = LayoutDetection(model_name="PP-DocLayout-L")


def build_hierarchy(layout_data, ocr_data):
  """
  Build a recursive hierarchy of layout boxes with nested OCR boxes.

  Args:
    layout_data: List of layout boxes from layout_model.predict(), each with 'coordinate' and 'type'
    ocr_data: List of OCR rec_boxes (coordinates only, as [x_min, y_min, x_max, y_max])

  Returns:
    Tuple of (root_nodes, orphaned_ocr_boxes) where:
      - root_nodes: List of HierarchyNode objects representing top-level layout boxes
      - orphaned_ocr_boxes: List of OCRBox objects not contained in any layout box
  """
  click.secho(f'    Building hierarchy from {len(layout_data)} layout boxes and {len(ocr_data)} OCR boxes', fg='cyan')

  # Convert raw layout data to LayoutBox dataclass instances
  layout_boxes = [
    LayoutBox(coordinates=box['coordinate'], layout_type=box.get('type', 'unknown'))
    for box in layout_data
  ]

  # Convert raw OCR data to OCRBox dataclass instances
  ocr_boxes = [OCRBox(coordinates=coord) for coord in ocr_data]

  # Sort layout boxes by area (smallest first) to facilitate nesting
  layout_boxes.sort(key=lambda box: (box.coordinates[2] - box.coordinates[0]) * (box.coordinates[3] - box.coordinates[1]))

  # Create a mapping from layout box to hierarchy node
  nodes_map = {i: HierarchyNode(layout_box=layout_boxes[i]) for i in range(len(layout_boxes))}

  # Build parent-child relationships by checking containment
  children_set = set()  # Track which nodes are children (to identify roots)

  for i, child_box in enumerate(layout_boxes):
    for j, parent_box in enumerate(layout_boxes):
      if i != j and is_box_contained(child_box.coordinates, parent_box.coordinates):
        # Check if parent_box is the immediate parent (smallest containing box)
        # by verifying no other box strictly contains child_box and is contained in parent_box
        is_immediate_parent = True
        for k, other_box in enumerate(layout_boxes):
          if k != i and k != j:
            if (is_box_contained(child_box.coordinates, other_box.coordinates) and
                is_box_contained(other_box.coordinates, parent_box.coordinates)):
              is_immediate_parent = False
              break

        if is_immediate_parent:
          nodes_map[j].children.append(nodes_map[i])
          children_set.add(i)
          break

  # Get root nodes (those not contained in any other box)
  root_nodes = [nodes_map[i] for i in range(len(layout_boxes)) if i not in children_set]
  click.secho(f'    Found {len(root_nodes)} root layout boxes', fg='cyan')

  # Assign OCR boxes to their deepest containing layout node
  orphaned_ocr = []

  for ocr_box in ocr_boxes:
    # Find the deepest (smallest) containing layout node
    deepest_node = None
    deepest_area = float('inf')

    def find_deepest_node(node):
      nonlocal deepest_node, deepest_area
      if is_box_contained(ocr_box.coordinates, node.layout_box.coordinates):
        area = (node.layout_box.coordinates[2] - node.layout_box.coordinates[0]) * \
               (node.layout_box.coordinates[3] - node.layout_box.coordinates[1])
        if area < deepest_area:
          deepest_area = area
          deepest_node = node

        # Recursively check children
        for child in node.children:
          find_deepest_node(child)

    # Check all root nodes and their descendants
    for root in root_nodes:
      find_deepest_node(root)

    if deepest_node:
      deepest_node.ocr_boxes.append(ocr_box)
    else:
      orphaned_ocr.append(ocr_box)

  click.secho(f'    Assigned {len(ocr_boxes) - len(orphaned_ocr)} OCR boxes to hierarchy, {len(orphaned_ocr)} orphaned', fg='cyan')

  return root_nodes, orphaned_ocr


def predict_ocr_and_layout(params):
  click.secho(f'    Running OCR Prediction on {params["scaled_img_path"]}', fg='cyan')
  params['results'] = ocr.predict(str(params["scaled_img_path"]))

  for i, points in enumerate(params["results"][0]['rec_boxes']):
    x_min, y_min, x_max, y_max = points

    scaled_coords = [x_min / params["scale"], y_min / params["scale"], x_max / params["scale"], y_max / params["scale"]]
    x_min, y_min, x_max, y_max =  map(int, scaled_coords)

    params["results"][0]['rec_boxes'][i] = [x_min, y_min, x_max, y_max]

  click.secho(f'    Running Layout Prediction on {params["scaled_img_path"]}', fg='cyan')
  params['layouts'] = layout_model.predict(str(params["scaled_img_path"]))

  for box in params["layouts"][0]['boxes']:
    x_min, y_min, x_max, y_max = box['coordinate']

    scaled_coords = [x_min / params["scale"], y_min / params["scale"], x_max / params["scale"], y_max / params["scale"]]
    x_min, y_min, x_max, y_max =  map(int, scaled_coords)

    box['coordinate'] = [x_min, y_min, x_max, y_max]


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

  for points in params['results'][0]['rec_boxes']:
    x_min, y_min, x_max, y_max = points

    # Create 4 corner points for the rectangle
    points = np.array([
      [x_min, y_min],
      [x_max, y_min],
      [x_max, y_max],
      [x_min, y_max]
    ], dtype=np.int32)

    cv2.polylines(params['img'], [points], True, (0, 255, 0), 2)  # Green for OCR

  for box in params["layouts"][0]['boxes']:
    x_min, y_min, x_max, y_max = box['coordinate']

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
  params['hierarchy'], params['orphaned_ocr'] = build_hierarchy(params['layouts'][0]['boxes'],
                                                                params['results'][0]['rec_boxes'])

  breakpoint()
  visualize_results(params)

