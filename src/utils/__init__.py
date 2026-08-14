import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import click
import requests
from paddleocr import LayoutDetection, PaddleOCR

from iiif_models import IIIFImageResource


class OCRBackend:
  def __init__(self, lang, **kwargs):
    self.ocr = PaddleOCR(
      use_doc_orientation_classify=True,
      use_doc_unwarping=False,
      use_textline_orientation=False,
      return_word_box=True,
      lang=lang,
      ocr_version='PP-OCRv5',
      **kwargs,
    )
    self.layout_model = LayoutDetection(model_name='PP-DocLayout-L')


def load_manifest(manifest_input) -> tuple[list[IIIFImageResource], str]:
  """
  Load a manifest from either a file path or a URI.

  Args:
    manifest_input: Either a file path (str/Path) or a URI string

  Returns:
    list of IIIFImageResource objects and a resource ID (str)

  Raises:
    click.ClickException: If the manifest cannot be loaded
  """
  # Try to load as a file first
  path = Path(manifest_input)
  if path.is_file():
    try:
      click.echo(f'Loading manifest from file: {manifest_input}')
      with open(path, 'r') as f:
        manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
      raise click.ClickException(f'Failed to load manifest from file: {e}')

  # Otherwise, try as URI
  else:
    try:
      click.echo(f'Loading manifest from URI: {manifest_input}')
      response = requests.get(manifest_input)
      response.raise_for_status()
      manifest = response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
      raise click.ClickException(f'Failed to load manifest from URI: {e}')

  click.secho('Successfully fetched JSON data.\n', fg='white')

  resource_id = str(uuid5(NAMESPACE_URL, manifest.get('id', manifest.get('@id'))))

  # Extract image resources
  images = []
  if '@id' in manifest:
    # IIIF Presentation API v2 structure
    for canvas in manifest.get('sequences', [{}])[0].get('canvases', []):
      if 'images' in canvas and canvas['images']:
        img_resource = IIIFImageResource.from_dict(canvas['images'][0]['resource'])
        images.append(img_resource)
  else:
    # IIIF Presentation API v3 structure
    for item in manifest.get('items', []):
      image = item['items'][0]['items'][0]['body'] # AnnotationPage -> Annotation -> Image
      img_resource = IIIFImageResource.from_dict(image)
      images.append(img_resource)

  return images, str(resource_id)
