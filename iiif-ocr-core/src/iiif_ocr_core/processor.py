import logging
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import requests
from numpy import array
from paddleocr import LayoutDetection, PaddleOCR
from PIL import Image, ImageDraw
from yattag import Doc, indent

from .manifest import IIIFImageResource, load_manifest
from .models import HOCR_MAPPINGS, Layout, Line, Word, contains, overlaps

logger = logging.getLogger(__name__)


class ProcessingError(RuntimeError):
  """Raised when a manifest cannot be processed."""


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


class OCRProcessor:
  """Process a IIIF manifest into hOCR files."""

  def __init__(
    self,
    manifest: str | Path,
    output_dir: Path = Path('downloads'),
    size: int = 1250,
    language: str = 'en',
    visualize: bool = False,
    gpu: bool = False,
    ocr_backend: OCRBackend | None = None,
    include_resource_id: bool = True,
  ):
    self.manifest = manifest
    self.output_dir = output_dir
    self.size = size
    self.language = language
    self.visualize = visualize
    self.gpu = gpu
    self.ocr_backend = ocr_backend
    self.include_resource_id = include_resource_id
    self.page = ''
    self.img_resource: IIIFImageResource | None = None
    self.img_path: Path | None = None
    self.img = None
    self.scale = 1
    self.ocr_results = []
    self.layout_results = []
    self.lines = []
    self.layouts = []
    self.hierarchy = []
    self.missing_lines = []
    self.ocr_visualization: bytes | None = None
    self.layout_visualization: bytes | None = None
    self.bboxes_visualization: bytes | None = None

  def __call__(self) -> list[Path]:
    return self.process()

  def process(self) -> list[Path]:
    images, resource_id = load_manifest(self.manifest)
    if not images:
      raise ProcessingError('No images found in the manifest.')

    output_dir = self.output_dir / resource_id if self.include_resource_id else self.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    self.output_dir = output_dir

    generated_files = []
    for index, image in enumerate(images):
      logger.info('[*] Processing page %s/%s...', index + 1, len(images))
      generated_files.append(self.process_page(image, index))

    return generated_files

  def process_page(self, image: IIIFImageResource, page_number: int) -> Path:
    """Process one manifest image, allowing callers to resume page-by-page."""
    self.output_dir.mkdir(parents=True, exist_ok=True)
    if self.ocr_backend is None:
      self.ocr_backend = OCRBackend(lang=self.language)

    self.page = f'page_{page_number}'
    self.img_resource = image
    self._process_page()
    return self.output_dir / f'{self.page}.html'

  def _process_page(self):
    """Process the current page using the processor's state."""
    self.ocr_visualization = None
    self.layout_visualization = None
    self.bboxes_visualization = None
    self.img_path = self.output_dir / f'{self.page}.{self.img_resource.get_format()}'

    if not self.img_path.exists():
      logger.warning('    Downloading image to %s', self.img_path)

      img_data = requests.get(self.img_resource.id).content
      self.img = Image.open(BytesIO(img_data))

      with open(self.img_path, 'wb') as output:
        output.write(img_data)
    else:
      self.img = Image.open(self.img_path)

    longest_side = max(self.img_resource.width, self.img_resource.height)
    self.scale = self.size / longest_side
    self.img.thumbnail((self.size, self.size))

    self._predict_ocr_and_layout()
    self.hierarchy = self._build_hierarchy()

    lines_in_layouts = [line for layout in self.layouts for line in layout.ocr_lines]
    line_ids = {id(line) for line in lines_in_layouts}

    self.missing_lines = [line for line in self.lines if id(line) not in line_ids]

    self._visualize_results()
    self._generate_hocr()

  def _build_hierarchy(self):
    logger.info('    Building hierarchy from %s layouts and %s OCR lines', len(self.layouts), len(self.lines))
    for line in self.lines:
      for layout in self.layouts:
        if (
          overlaps(line.coordinates, layout.coordinates)
          or contains(line.coordinates, layout.coordinates)
          or contains(layout.coordinates, line.coordinates)
        ):
          layout.ocr_lines.append(line)
          break
    return self.layouts

  def _predict_ocr_and_layout(self):
    logger.info('    Running OCR Prediction')

    self.img = self.img.convert('RGB')
    results = self.ocr_backend.ocr.predict(array(self.img)[:, :, ::-1])

    angle = results[0]['doc_preprocessor_res']['angle']
    if angle in (90, 180, 270):
      logger.info('    Image rotated %s degrees', angle)
      self.img = self.img.rotate(angle, expand=True)

    width, height = self.img.size
    self.ocr_results = results
    lines = results[0]

    self.lines = []
    for line_box, word_boxes, word_texts in zip(lines['rec_boxes'], lines['text_word_boxes'], lines['text_word']):
      line_coords, word_coords = self._rotate_coordinates(
        angle, line_box.tolist(), [box for box in word_boxes.tolist()], width, height
      )

      self.lines.append(
        Line(
          coordinates=[coord / self.scale for coord in line_coords],
          words=[
            Word(coordinates=[coord / self.scale for coord in coords], text=text)
            for coords, text in zip(word_coords, word_texts)
          ],
        )
      )

    logger.info('    Running Layout Prediction')

    results = self.ocr_backend.layout_model.predict(array(self.img)[:, :, ::-1])
    self.layout_results = results

    self.layouts = []
    for box in results[0]['boxes']:
      coords, _ = self._rotate_coordinates(angle, box['coordinate'], [], width, height)
      self.layouts.append(Layout(layout_type=box['label'], coordinates=[coord / self.scale for coord in coords]))

  @staticmethod
  def _rotate_coordinates(angle, coords, word_coords, width, height):
    def rotate(values):
      x1, y1, x2, y2 = values
      if angle == 90:
        return [height - y2, x1, height - y1, x2]
      if angle == 180:
        return [width - x2, height - y2, width - x1, height - y1]
      if angle == 270:
        return [y1, width - x2, y2, width - x1]
      return values

    return rotate(coords), [rotate(values) for values in word_coords]

  def _visualize_results(self):
    if not self.visualize:
      return

    logger.warning('    Saving OCR visualizations in %s', self.output_dir)
    image_extension = self.img_resource.get_format().lower()
    pillow_format = Image.registered_extensions().get(f'.{image_extension}')
    if pillow_format is None:
      raise ProcessingError(f'Unsupported image format: {self.img_resource.get_format()}')

    self.ocr_visualization = self._results_to_bytes(self.ocr_results, image_extension)

    ocr_path = self.output_dir / f'{self.page}_ocr_visualization.{image_extension}'
    ocr_path.write_bytes(self.ocr_visualization)

    logger.warning('    Saving layout visualizations in %s', self.output_dir)
    self.layout_visualization = self._results_to_bytes(self.layout_results, image_extension)

    layout_path = self.output_dir / f'{self.page}_layout_visualization.{image_extension}'
    layout_path.write_bytes(self.layout_visualization)

    self.img = Image.open(self.img_path).convert('RGB')
    drawing = ImageDraw.Draw(self.img)

    for line in self.lines:
      drawing.rectangle(xy=line.coordinates, outline='green', width=2, fill=None)

    for layout in self.layouts:
      drawing.rectangle(xy=layout.coordinates, outline='blue', width=2, fill=None)

    bboxes_buffer = BytesIO()
    self.img.save(bboxes_buffer, format=pillow_format)
    self.bboxes_visualization = bboxes_buffer.getvalue()

    bboxes_path = self.output_dir / f'{self.page}_bboxes.{image_extension}'
    bboxes_path.write_bytes(self.bboxes_visualization)

    logger.info('    Saved image with bboxes to %s', bboxes_path)

  @staticmethod
  def _results_to_bytes(results, image_format: str) -> bytes:
    """Convert PaddleOCR's path-based visualization output into image bytes."""
    with TemporaryDirectory() as directory:
      path = Path(directory) / f'visualization.{image_format}'

      for result in results:
        result.save_to_img(path)

      return path.read_bytes()

  def _generate_hocr(self):
    hocr_path = self.output_dir / f'{self.page}.html'
    image_path = self.output_dir / f'{self.page}.{self.img_resource.get_format()}'

    doc, tag, text, line = Doc().ttl()
    doc.asis('<?xml version="1.0" encoding="UTF-8"?>')
    doc.asis(
      '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN" '
      '"http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">'
    )

    with tag('html', xmlns='http://www.w3.org/1999/xhtml', **{'xml:lang': 'en'}, lang='en'), tag('head'):
      line('title', '')
      doc.stag('meta', **{'http-equiv': 'Content-Type'}, content='text/html;charset=utf-8')
      doc.stag('meta', name='ocr-system', content=f'paddleocr {version("paddleocr")}')
      doc.stag('meta', name='ocr-capabilities', content='ocr_page ocr_carea ocr_par ocr_line ocrx_word')

    with tag('body'), tag(
      'div',
      id=self.page,
      klass='ocr_page',
      title=f'image "{image_path}"; bbox 0 0 {self.img_resource.width} {self.img_resource.height}',
    ):
      for layout in self.layouts:
        with tag(
          'div',
          klass=HOCR_MAPPINGS[layout.layout_type],
          title=f'bbox {int(layout.coordinates[0])} {int(layout.coordinates[1])} '
          f'{int(layout.coordinates[2])} {int(layout.coordinates[3])}',
        ):
          for ocr_line in layout.ocr_lines:
            self._write_hocr_line(tag, text, doc, ocr_line)

      for ocr_line in self.missing_lines:
        self._write_hocr_line(tag, text, doc, ocr_line)

    with open(hocr_path, 'w') as output:
      output.write(indent(doc.getvalue()))

    logger.info('    Generated hOCR file at %s', hocr_path)

  @staticmethod
  def _write_hocr_line(tag, text, doc, ocr_line):
    with tag(
      'span',
      klass='ocr_line',
      title=f'bbox {int(ocr_line.coordinates[0])} {int(ocr_line.coordinates[1])} '
      f'{int(ocr_line.coordinates[2])} {int(ocr_line.coordinates[3])}',
    ):
      for word in ocr_line.words:
        with tag(
          'span',
          klass='ocrx_word',
          title=f'bbox {int(word.coordinates[0])} {int(word.coordinates[1])} '
          f'{int(word.coordinates[2])} {int(word.coordinates[3])}',
        ):
          if word.text == ' ':
            doc.asis('&nbsp;')
          else:
            text(word.text)
