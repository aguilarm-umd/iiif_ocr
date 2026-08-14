from pathlib import Path
from traceback import format_exc

import click

from process import process
from utils import OCRBackend, load_manifest


@click.command()
@click.argument('manifest', type=str, metavar='<URL or local path>')
@click.option('--language', type=str, default='en', help='Language for OCR engine to detect, see languages.md')
@click.option('--visualize', is_flag=True, default=False, help='Visualize Bounding Boxes')
@click.option('--gpu', is_flag=True, default=False, help='Use GPU when using OCR engine')
def main(**kwargs):
  """
  Generate an hOCR file from a IIIF Manifest
  """
  params = kwargs

  params['ocr_backend'] = OCRBackend(lang=params['language'])

  try:
    images, resource_id = load_manifest(params['manifest'])
    click.secho('Successfully loaded manifest.\n', fg='white')
  except click.ClickException as e:
    click.secho(str(e), fg='red')
    ctx = click.get_current_context()
    ctx.exit(code=1)

  if resource_id is None:
    click.secho('Could not extract Resource ID', fg='red')
    ctx = click.get_current_context()
    ctx.exit(code=1)

  output_dir = Path('downloads') / resource_id
  click.secho(f'Storing content in {output_dir}\n', fg='white')

  output_dir.mkdir(parents=True, exist_ok=True)
  params['output_dir'] = output_dir

  try:
    if not images:
      raise click.ClickException('No images found in the manifest.')

    for i, image in enumerate(images):
      click.secho(f'[*] Processing page {i + 1}/{len(images)}...', fg='white')

      params['page'] = f'page_{i}'
      params['img_resource'] = image

      process(params)

    click.secho('\nDone!', fg='green', bold=True)

  except Exception:
    click.secho(format_exc(), fg='red')
    ctx = click.get_current_context()
    ctx.exit(code=1)


if __name__ == '__main__':
  main()
