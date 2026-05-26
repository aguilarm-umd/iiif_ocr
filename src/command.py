from pathlib import Path
from traceback import format_exc

import click

from iiif_models import IIIFImageResource
from process import process
from utils import extract_uuid, load_manifest


@click.command()
@click.argument('manifest', type=str)
@click.option('--visualize', is_flag=True, default=False, help='Visualize Bounding Boxes')
@click.option('--gpu', is_flag=True, default=False, help='Use GPU when using OCR engine')
def main(**kwargs):
  """
  Generate an hOCR file from a IIIF Manifest
  """
  params = kwargs

  try:
    manifest_data = load_manifest(params["manifest"])
    click.secho('Successfully loaded manifest.\n', fg='white')
  except click.ClickException as e:
    click.secho(str(e), fg='red')
    ctx = click.get_current_context()
    ctx.exit(code=1)

  resource_id = extract_uuid(manifest_data['@id'])

  if resource_id is None:
    click.secho('Could not extract Resource ID', fg='red')
    ctx = click.get_current_context()
    ctx.exit(code=1)

  output_dir = Path('downloads') / resource_id
  output_dir.mkdir(parents=True, exist_ok=True)
  params['output_dir'] = output_dir

  try:
    manifest = manifest_data
    click.secho('Successfully fetched JSON data.\n', fg='white')

    sequences = manifest.get('sequences', [])
    canvases = sequences[0].get('canvases', []) if sequences else manifest.get('items', [])

    if not canvases:
      raise click.ClickException('No canvases found in the manifest.')

    for i, canvas in enumerate(canvases):
      click.secho(f'[*] Processing page {i + 1}/{len(canvases)}...', fg='white')

      try:
        img_resource = IIIFImageResource.from_dict(canvas['images'][0]['resource'])

      except (KeyError, IndexError):
        click.secho(f'Skip: Could not find image for canvas {i}', fg='bright_yellow')
        continue

      params['page'] = f'page_{i}'
      params['img_resource'] = img_resource

      process(params)

    click.secho('\nDone!', fg='green', bold=True)

  except Exception:
    click.secho(format_exc(), fg='red')
    ctx = click.get_current_context()
    ctx.exit(code=1)


if __name__ == '__main__':
  main()
