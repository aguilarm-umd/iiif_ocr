import click
import re
import requests
import numpy as np
import cv2

from iiif_ocr import process
from urllib.parse import urlparse
from pathlib import Path


def extract_uuid(path):
  """
  Finds a UUID (8-4-4-4-12 pattern) in a string.
  Example: 162922a8-1dbb-46ee-9425-a10f3665fe7d
  """
  match = re.search(r'[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}', path)
  return match.group(0) if match else None

def scale_image(img_data, max_dim=4000, quality=95):
  img_array = np.frombuffer(img_data, np.uint8)
  img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

  h, w = img.shape[:2]
  longest_side = max(h, w)

  if longest_side <= max_dim:
    click.secho(f"[*] Kept original size: {w}x{h}", fg='cyan')
    return img_data
  
  scale = max_dim / longest_side
  new_w = int(w * scale)
  new_h = int(h * scale)
    
  # Resize using INTER_AREA (best for downscaling)
  img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
  click.secho(f"[*] Downscaled: {w}x{h} -> {new_w}x{new_h}", fg='cyan')

  _, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
  return buffer.tobytes()


@click.command()
@click.argument('uri', type=urlparse)
@click.option('--visualize', is_flag=True, default=False, help='Vizualize Bounding Boxes')
@click.option('--gpu', is_flag=True, default=False, help='Use GPU when using OCR engine')
@click.option(
  '--model',
  '-m',
  type=click.Choice(['hjd', 'nn'], case_sensitive=False),
  default='nn',
  help='Pre-trained model provided as options by Layout Parser. Supports HJDataset and Newspaper Navigator',
)
def main(uri, visualize, model, gpu):
  """
  Generate an hOCR file from a IIIF Manifest
  """
  click.echo(f'Retrieving Manifest from: {uri.geturl()}')

  resource_id = extract_uuid(uri.path)

  if resource_id is None:
    click.secho('Could not extract Resource ID', fg='red')
    ctx = click.get_current_context()
    ctx.exit(code=1)

  output_dir = Path('downloads') / resource_id
  output_dir.mkdir(parents=True, exist_ok=True)

  try:
    response = requests.get(uri.geturl(), timeout=30)
    response.raise_for_status()

    manifest = response.json()
    click.secho('Successfully fetched JSON data.\n', fg='cyan')

    sequences = manifest.get('sequences', [])
    canvases = sequences[0].get('canvases', []) if sequences else manifest.get('items', [])

    if not canvases:
      raise click.ClickException('No canvases found in the manifest.')

    for i, canvas in enumerate(canvases):
      click.secho(f'[*] Processing page {i + 1}/{len(canvases)}...', fg='cyan')
      try:
        img_resource = canvas['images'][0]['resource']
        img_url = img_resource['@id']

        image_ext = 'jpg' if 'image/jpeg' in img_resource.get('format', '') else 'png'
        image_path = output_dir / f'page_{i}.{image_ext}'
        hocr_path = output_dir / f'page_{i}.html'

        if not image_path.exists():
          img_data = requests.get(img_url).content
          img_data = scale_image(img_data)

          with open(image_path, 'wb') as f:
            f.write(img_data)

        if not hocr_path.exists():
          hocr_data = process(image_path, visualize, model, gpu)

        #   with open(hocr_path, 'wb') as f:
        #     f.write(hocr_data)

      except (KeyError, IndexError):
        click.secho(f'Skip: Could not find image for canvas {i}', fg='yellow')

    click.secho('\nDone!', fg='green', bold=True)

  except Exception as e:
    click.secho(e, fg='red')
    ctx = click.get_current_context()
    ctx.exit(code=1)


if __name__ == '__main__':
  main()
