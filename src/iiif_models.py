from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ImageProfile:
  """IIIF Image Service profile with supported features."""
  formats: List[str]
  qualities: List[str]
  supports: List[str]


@dataclass
class ImageService:
  """IIIF Image Service descriptor."""
  context: str = "http://iiif.io/api/image/2/context.json"
  id: str = ""  # @id in JSON
  profile: List[Any] = field(default_factory=list)  # Can be string or ImageProfile dict

  def to_dict(self) -> Dict[str, Any]:
    """Convert to JSON-serializable dict with @id and @context."""
    data = asdict(self)
    data["@context"] = data.pop("context")
    data["@id"] = data.pop("id")
    return data


@dataclass
class IIIFImageResource:
  """
  IIIF Image Resource (dctypes:Image) with embedded image service.

  Represents the "resource" object in IIIF manifests.
  """
  id: str  # @id in JSON
  type: str = "dctypes:Image"  # @type in JSON
  format: str = "image/jpeg"
  height: int = 0
  width: int = 0
  service: Optional[ImageService] = None

  def to_dict(self) -> Dict[str, Any]:
    """Convert to JSON-serializable dict with IIIF @ prefixes."""
    data = {
      "@id": self.id,
      "@type": self.type,
      "format": self.format,
      "height": self.height,
      "width": self.width,
    }
    if self.service:
      data["service"] = self.service.to_dict()
    return data

  def get_format(self) -> str:
    """Extract file extension from format string."""
    return self.format.split("/")[-1]  # e.g., "jpeg" from "image/jpeg"

  @classmethod
  def from_dict(cls, data: Dict[str, Any]) -> "IIIFImageResource":
    """Create instance from IIIF JSON dict."""
    service_data = data.get("service")
    service = None
    if service_data:
      service = ImageService(
        context=service_data.get("@context", "http://iiif.io/api/image/2/context.json"),
        id=service_data.get("@id", ""),
        profile=service_data.get("profile", [])
      )

    return cls(
      id=data.get("@id", ""),
      type=data.get("@type", "dctypes:Image"),
      format=data.get("format", "image/jpeg"),
      height=data.get("height", 0),
      width=data.get("width", 0),
      service=service
    )
