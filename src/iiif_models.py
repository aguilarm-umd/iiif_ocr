from dataclasses import dataclass
from typing import Any, Dict


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
