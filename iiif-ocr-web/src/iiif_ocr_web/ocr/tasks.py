from pathlib import Path

from django.conf import settings
from django.db import transaction
from django.tasks import task
from django.utils import timezone
from django_absurd import get_absurd_context
from iiif_ocr_core.manifest import IIIFImageResource, load_manifest
from iiif_ocr_core.processor import OCRProcessor

from .models import OCRJob, OCRJobStatus, Page, PageStatus


@task
def run_ocr_job(job_id: int) -> str:
  job = OCRJob.objects.select_related('manifest').get(pk=job_id)
  context = get_absurd_context()

  job.status = OCRJobStatus.PROCESSING
  if job.started_at is None:
    job.started_at = timezone.now()
  job.completed_at = None
  job.error_message = ''

  job.save(update_fields=['status', 'started_at', 'completed_at', 'error_message', 'updated_at'])

  image_data = context.step(
    'load-manifest',
    lambda: [
      {
        'id': image.id,
        'type': image.type,
        'format': image.format,
        'height': image.height,
        'width': image.width,
      }
      for image in load_manifest(job.manifest.url)[0]
    ],
  )
  if not image_data:
    raise ValueError('No usable IIIF image resources were found in the manifest.')

  images = [IIIFImageResource(**data) for data in image_data]
  base_dir = Path(settings.MEDIA_ROOT) / str(job.manifest_id) / str(job.number)
  base_dir.mkdir(parents=True, exist_ok=True)
  processor = OCRProcessor(
    manifest=job.manifest.url,
    output_dir=base_dir,
    size=job.image_size,
    language=job.language,
    visualize=job.visualize,
    gpu=False,
    include_resource_id=False,
  )

  with transaction.atomic():
    pages = []
    for index, image in enumerate(images):
      page, _ = Page.objects.get_or_create(
        job=job,
        page_number=index,
        defaults={
          'source_url': image.id,
          'width': image.width,
          'height': image.height,
          'status': PageStatus.PENDING,
        },
      )
      page.source_url = image.id
      page.width = image.width
      page.height = image.height
      page.save(update_fields=['source_url', 'width', 'height'])
      pages.append(page)

  for page, image in zip(pages, images):
    def process_page():
      page.status = PageStatus.PROCESSING
      page.error_message = ''
      page.save(update_fields=['status', 'error_message'])

      job.current_page = page.page_number + 1
      job.save(update_fields=['current_page', 'updated_at'])

      try:
        output_path = processor.process_page(image, page.page_number)
      except Exception as exc:
        page.status = PageStatus.FAILED
        page.error_message = str(exc)
        page.save(update_fields=['status', 'error_message'])
        job.status = OCRJobStatus.FAILED
        job.error_message = str(exc)
        job.completed_at = timezone.now()
        job.save(update_fields=['status', 'error_message', 'completed_at', 'updated_at'])
        raise

      page.status = PageStatus.COMPLETED
      page.hocr = output_path.read_text()
      page.save(update_fields=['status', 'hocr'])
      return str(output_path)

    context.step(f'process-page-{page.page_number}', process_page)

  job.status = OCRJobStatus.COMPLETED
  job.completed_at = timezone.now()
  job.current_page = len(images)
  job.save(update_fields=['status', 'current_page', 'completed_at', 'updated_at'])

  return f'Processed {len(images)} page(s) for manifest {job.manifest_id}'
