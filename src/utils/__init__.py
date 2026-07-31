import json
import re
from pathlib import Path

import click
import requests
from paddleocr import LayoutDetection, PaddleOCR


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


def load_manifest(manifest_input):
  """
  Load a manifest from either a file path or a URI.

  Args:
    manifest_input: Either a file path (str/Path) or a URI string

  Returns:
    dict: The parsed manifest JSON

  Raises:
    click.ClickException: If the manifest cannot be loaded
  """
  # Try to load as a file first
  path = Path(manifest_input)
  if path.exists() and path.is_file():
    try:
      click.echo(f'Loading manifest from file: {manifest_input}')
      with open(path, 'r') as f:
        return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
      raise click.ClickException(f'Failed to load manifest from file: {e}')

  # Otherwise, try as URI
  try:
    click.echo(f'Loading manifest from URI: {manifest_input}')
    response = requests.get(manifest_input)
    response.raise_for_status()
    return response.json()
  except (requests.RequestException, json.JSONDecodeError) as e:
    raise click.ClickException(f'Failed to load manifest from URI: {e}')


def extract_uuid(path):
  """
  Finds a UUID (8-4-4-4-12 pattern) in a string.
  Example: 162922a8-1dbb-46ee-9425-a10f3665fe7d
  """
  match = re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', path)
  return match.group(0) if match else None
