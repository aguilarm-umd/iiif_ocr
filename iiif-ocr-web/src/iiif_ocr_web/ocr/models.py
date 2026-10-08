import uuid

from django.db import models, transaction


class LanguageCode(models.TextChoices):
  ABAZA = 'abq', 'Abaza'
  AFRIKAANS = 'af', 'Afrikaans'
  OLD_ENGLISH = 'ang', 'Old English'
  ARABIC = 'ar', 'Arabic'
  AVARIC = 'ava', 'Avaric'
  AZERBAIJANI = 'az', 'Azerbaijani'
  BELARUSIAN = 'be', 'Belarusian'
  BULGARIAN = 'bg', 'Bulgarian'
  HARYANVI = 'bgc', 'Haryanvi'
  BIHARI = 'bh', 'Bihari'
  BHOJPURI = 'bho', 'Bhojpuri'
  BOSNIAN = 'bs', 'Bosnian'
  CHINESE = 'ch', 'Chinese (Simplified)'
  CHECHEN = 'che', 'Chechen'
  CHINESE_TRADITIONAL = 'chinese_cht', 'Chinese (Traditional)'
  CZECH = 'cs', 'Czech'
  WELSH = 'cy', 'Welsh'
  DANISH = 'da', 'Danish'
  DARGWA = 'dar', 'Dargwa'
  GERMAN = 'de', 'German'
  ENGLISH = 'en', 'English'
  SPANISH = 'es', 'Spanish'
  ESTONIAN = 'et', 'Estonian'
  PERSIAN = 'fa', 'Persian'
  FRENCH = 'fr', 'French'
  IRISH = 'ga', 'Irish'
  KONKANI = 'gom', 'Konkani'
  HINDI = 'hi', 'Hindi'
  CROATIAN = 'hr', 'Croatian'
  HUNGARIAN = 'hu', 'Hungarian'
  INDONESIAN = 'id', 'Indonesian'
  INGUSH = 'inh', 'Ingush'
  ICELANDIC = 'is', 'Icelandic'
  ITALIAN = 'it', 'Italian'
  JAPANESE = 'japan', 'Japanese'
  GEORGIAN = 'ka', 'Georgian'
  KABARDIAN = 'kbd', 'Kabardian'
  KOREAN = 'korean', 'Korean'
  KURDISH = 'ku', 'Kurdish'
  LATIN = 'la', 'Latin'
  LAK = 'lbe', 'Lak'
  LEZGHIAN = 'lez', 'Lezghian'
  LITHUANIAN = 'lt', 'Lithuanian'
  LATVIAN = 'lv', 'Latvian'
  MAGAHI = 'mah', 'Magahi'
  MAITHILI = 'mai', 'Maithili'
  MAORI = 'mi', 'Maori'
  MONGOLIAN = 'mn', 'Mongolian'
  MARATHI = 'mr', 'Marathi'
  MALAY = 'ms', 'Malay'
  MALTESE = 'mt', 'Maltese'
  NEPALI = 'ne', 'Nepali'
  NEWARI = 'new', 'Newari'
  DUTCH = 'nl', 'Dutch'
  NORWEGIAN = 'no', 'Norwegian'
  OCCITAN = 'oc', 'Occitan'
  PALI = 'pi', 'Pali'
  POLISH = 'pl', 'Polish'
  PORTUGUESE = 'pt', 'Portuguese'
  ROMANIAN = 'ro', 'Romanian'
  SERBIAN_CYRILLIC = 'rs_cyrillic', 'Serbian (Cyrillic)'
  SERBIAN_LATIN = 'rs_latin', 'Serbian (Latin)'
  RUSSIAN = 'ru', 'Russian'
  SANSKRIT = 'sa', 'Sanskrit'
  SADRI = 'sck', 'Sadri'
  SLOVAK = 'sk', 'Slovak'
  SLOVENIAN = 'sl', 'Slovenian'
  ALBANIAN = 'sq', 'Albanian'
  SWEDISH = 'sv', 'Swedish'
  SWAHILI = 'sw', 'Swahili'
  TABASSARAN = 'tab', 'Tabassaran'
  TAMIL = 'ta', 'Tamil'
  TELUGU = 'te', 'Telugu'
  TAGALOG = 'tl', 'Tagalog'
  TURKISH = 'tr', 'Turkish'
  UYGHUR = 'ug', 'Uyghur'
  UKRAINIAN = 'uk', 'Ukrainian'
  URDU = 'ur', 'Urdu'
  UZBEK = 'uz', 'Uzbek'
  VIETNAMESE = 'vi', 'Vietnamese'


class ImageSize(models.IntegerChoices):
  SMALL = 625, 'Small'
  MEDIUM = 1250, 'Medium'
  LARGE = 2500, 'Large'


class ManifestStatus(models.TextChoices):
  AVAILABLE = 'available', 'Available'
  INVALID = 'invalid', 'Invalid'


class OCRJobStatus(models.TextChoices):
  PENDING = 'pending', 'Pending'
  QUEUED = 'queued', 'Queued'
  PROCESSING = 'processing', 'Processing'
  COMPLETED = 'completed', 'Completed'
  FAILED = 'failed', 'Failed'


class PageStatus(models.TextChoices):
  PENDING = 'pending', 'Pending'
  PROCESSING = 'processing', 'Processing'
  COMPLETED = 'completed', 'Completed'
  FAILED = 'failed', 'Failed'


def ocr_upload_path(instance, filename):
  return (
    f'{instance.job.manifest_id}/{instance.job.number}/'
    f'page_{instance.page_number}/{filename}'
  )


class Manifest(models.Model):
  """A stable IIIF manifest that can be submitted for OCR repeatedly."""

  id = models.UUIDField(primary_key=True, editable=False)
  url = models.URLField()
  status = models.CharField(
    max_length=20,
    choices=ManifestStatus.choices,
    default=ManifestStatus.AVAILABLE,
  )
  error_message = models.TextField(blank=True)
  created_at = models.DateTimeField(auto_now_add=True)
  updated_at = models.DateTimeField(auto_now=True)

  def __str__(self):
    return str(self.id)


class OCRJob(models.Model):
  """One OCR submission for a manifest, including its configuration and results."""

  number = models.PositiveIntegerField(editable=False)
  manifest = models.ForeignKey(Manifest, related_name='jobs', on_delete=models.CASCADE)
  language = models.CharField(
    max_length=20,
    choices=LanguageCode.choices,
    default=LanguageCode.ENGLISH,
  )
  image_size = models.PositiveSmallIntegerField(
    choices=ImageSize.choices,
    default=ImageSize.MEDIUM,
  )
  visualize = models.BooleanField(default=False)
  status = models.CharField(
    max_length=20,
    choices=OCRJobStatus.choices,
    default=OCRJobStatus.PENDING,
  )
  current_page = models.PositiveIntegerField(null=True, blank=True)
  error_message = models.TextField(blank=True)
  created_at = models.DateTimeField(auto_now_add=True)
  updated_at = models.DateTimeField(auto_now=True)
  started_at = models.DateTimeField(null=True, blank=True)
  completed_at = models.DateTimeField(null=True, blank=True)

  def save(self, *args, **kwargs):
    if self.number is None:
      with transaction.atomic():
        Manifest.objects.select_for_update().get(pk=self.manifest_id)
        last_job = OCRJob.objects.filter(manifest=self.manifest).order_by('-number').first()
        self.number = 1 if last_job is None else last_job.number + 1
        super().save(*args, **kwargs)
      return
    super().save(*args, **kwargs)

  def __str__(self):
    return f'{self.manifest_id} job {self.number}'

  class Meta:
    constraints = [
      models.UniqueConstraint(
        fields=['manifest', 'number'],
        name='unique_manifest_job_number',
      ),
    ]


class Page(models.Model):
  job = models.ForeignKey(OCRJob, related_name='pages', on_delete=models.CASCADE)
  page_number = models.PositiveIntegerField()
  source_url = models.URLField()
  width = models.PositiveIntegerField()
  height = models.PositiveIntegerField()
  status = models.CharField(
    max_length=20,
    choices=PageStatus.choices,
    default=PageStatus.PENDING,
  )
  hocr = models.TextField(blank=True)
  error_message = models.TextField(blank=True)
  source_image = models.FileField(upload_to=ocr_upload_path, blank=True)
  ocr_visualization = models.FileField(upload_to=ocr_upload_path, blank=True)
  layout_visualization = models.FileField(upload_to=ocr_upload_path, blank=True)
  bboxes_visualization = models.FileField(upload_to=ocr_upload_path, blank=True)

  class Meta:
    constraints = [
      models.UniqueConstraint(
        fields=['job', 'page_number'],
        name='unique_job_page_number',
      ),
    ]
    ordering = ['job', 'page_number']

  def __str__(self):
    return f'{self.job_id} page {self.page_number}'
