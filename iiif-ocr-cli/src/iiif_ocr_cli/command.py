import logging
from traceback import format_exc

import click

from iiif_ocr_core.processor import OCRProcessor, ProcessingError
from iiif_ocr_core.manifest import ManifestError


class ClickLogHandler(logging.Handler):
  _colors = {
    logging.INFO: 'cyan',
    logging.WARNING: 'yellow',
    logging.ERROR: 'red',
    logging.CRITICAL: 'red',
  }

  def emit(self, record):
    try:
      click.secho(self.format(record), fg=self._colors.get(record.levelno, 'white'))
    except Exception:
      self.handleError(record)


def configure_logging():
  logger = logging.getLogger('iiif_ocr_core')
  logger.handlers.clear()
  logger.addHandler(ClickLogHandler())
  logger.setLevel(logging.INFO)
  logger.propagate = False


@click.command()
@click.argument('manifest', type=str, metavar='<URL or local path>')
@click.option(
    '--size', 
    type=click.Choice(['small', 'medium', 'large'], case_sensitive=False),
    default='medium',
    show_default=True,
    help='Select the size of the image when downscaling. The options correspond to a downscaled image of 625, 1250, or 2500 pixels respectively'
)
@click.option('--language', type=str, default='en', help='Language for OCR engine to detect, see languages.md')
@click.option('--visualize', is_flag=True, default=False, help='Visualize Bounding Boxes')
@click.option('--gpu', is_flag=True, default=False, help='Use GPU when using OCR engine')
def main(**kwargs):
  """
  Generate an hOCR file from a IIIF Manifest
  """
  configure_logging()
  params = kwargs

  match params['size']:
    case 'small':
      params['size'] = 625
    case 'medium':
      params['size'] = 1250
    case 'large':
      params['size'] = 2500

  try:
    processor = OCRProcessor(
      manifest=kwargs['manifest'],
      size=kwargs['size'],
      language=kwargs['language'],
      visualize=kwargs['visualize'],
      gpu=kwargs['gpu'],
    )
    generated_files = processor()
    click.secho(f'\nDone! Generated {len(generated_files)} hOCR file(s).', fg='green', bold=True)
  except (ManifestError, ProcessingError) as e:
    click.secho(str(e), fg='red')
    ctx = click.get_current_context()
    ctx.exit(code=1)
  except Exception:
    click.secho(format_exc(), fg='red')
    ctx = click.get_current_context()
    ctx.exit(code=1)


if __name__ == '__main__':
  main()
