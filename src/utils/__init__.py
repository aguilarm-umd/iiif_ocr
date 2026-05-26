import json
import re
from pathlib import Path

import click
import cv2
import numpy as np
import requests


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
    response = requests.get(manifest_input, timeout=30)
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


def scale_image(img_data, max_dim=2500, quality=95):
  img_array = np.frombuffer(img_data, np.uint8)
  img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

  h, w = img.shape[:2]
  longest_side = max(h, w)

  if longest_side <= max_dim:
    click.secho(f"[*] Kept original size: {w}x{h}", fg='cyan')
    return img_data, 1.0

  scale = max_dim / longest_side
  new_w = int(w * scale)
  new_h = int(h * scale)

  # Resize using INTER_AREA (best for downscaling)
  img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
  click.secho(f"[*] Downscaled: {w}x{h} -> {new_w}x{new_h}", fg='cyan')

  _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
  return buffer.tobytes(), scale
