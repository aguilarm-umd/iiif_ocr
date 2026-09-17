import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict
from uuid import NAMESPACE_URL, uuid5

import requests


logger = logging.getLogger(__name__)


@dataclass
class IIIFImageResource:
  """
  IIIF Image Resource (dctypes:Image) with embedded image service.

  Represents the "resource" object in IIIF manifests.
  """

  id: str
  type: str
  format: str
  height: int
  width: int

  def get_format(self) -> str:
    """Extract file extension from format string."""
    return self.format.split('/')[-1]  # e.g., "jpeg" from "image/jpeg"

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> 'IIIFImageResource':
    """Create instance from IIIF JSON dict."""

    return cls(
      id=data.get('@id') if data.get('@id') else data.get('id'),
      type=data.get('@type') if data.get('@type') else data.get('type'),
      format=data.get('format'),
      height=data.get('height'),
      width=data.get('width'),
    )


class ManifestError(ValueError):
  """Raised when a IIIF manifest cannot be loaded."""


def load_manifest(manifest_input) -> tuple[list[IIIFImageResource], str]:
  """
  Load a manifest from either a file path or a URI.

  Args:
    manifest_input: Either a file path (str/Path) or a URI string

  Returns:
    list of IIIFImageResource objects and a resource ID (str)

  Raises:
    ManifestError: If the manifest cannot be loaded
  """
  path = Path(manifest_input)
  if path.is_file():
    try:
      logger.info('Loading manifest from file: %s', manifest_input)
      with open(path, 'r') as f:
        manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
      raise ManifestError(f'Failed to load manifest from file: {e}') from e
  else:
    try:
      logger.info('Loading manifest from URI: %s', manifest_input)
      response = requests.get(manifest_input)
      response.raise_for_status()
      manifest = response.json()
    except (requests.RequestException, json.JSONDecodeError) as e:
      raise ManifestError(f'Failed to load manifest from URI: {e}') from e

  logger.info('Successfully fetched JSON data.')
  resource_id = str(uuid5(NAMESPACE_URL, manifest.get('id', manifest.get('@id'))))

  images = []
  if '@id' in manifest:
    for canvas in manifest.get('sequences', [{}])[0].get('canvases', []):
      if 'images' in canvas and canvas['images']:
        images.append(IIIFImageResource.from_dict(canvas['images'][0]['resource']))
  else:
    for item in manifest.get('items', []):
      image = item['items'][0]['items'][0]['body']
      images.append(IIIFImageResource.from_dict(image))

  return images, str(resource_id)
