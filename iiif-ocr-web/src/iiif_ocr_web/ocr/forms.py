from django import forms

from iiif_ocr_core.manifest import ManifestError, load_manifest

from .models import ImageSize, LanguageCode, OCRJob
from .tasks import run_ocr_job


class OCRJobForm(forms.ModelForm):
  class Meta:
    model = OCRJob
    fields = ('language', 'image_size', 'visualize')


class ManifestSubmissionForm(forms.Form):
  url = forms.URLField(
    label='IIIF manifest URL',
    widget=forms.URLInput(attrs={'placeholder': 'https://your-library.example/manifest.json'}),
  )
  language = forms.ChoiceField(choices=LanguageCode.choices, initial=LanguageCode.ENGLISH)
  image_size = forms.TypedChoiceField(
    choices=ImageSize.choices,
    initial=ImageSize.MEDIUM,
    coerce=int,
  )
  visualize = forms.BooleanField(required=False, label='Generate visualizations')

  def clean_url(self):
    url = self.cleaned_data['url']
    try:
      images, manifest_id = load_manifest(url)
    except (ManifestError, TypeError, ValueError, KeyError, IndexError) as exc:
      raise forms.ValidationError(f'Unable to load IIIF manifest: {exc}') from exc

    if not images:
      raise forms.ValidationError('The IIIF manifest does not contain any usable images.')

    self.cleaned_data['manifest_id'] = manifest_id
    return url

  def save(self):
    from uuid import UUID

    from .models import Manifest

    url = self.cleaned_data['url']
    manifest, _ = Manifest.objects.get_or_create(
      id=UUID(self.cleaned_data['manifest_id']),
      defaults={'url': url},
    )
    job = OCRJob.objects.create(
      manifest=manifest,
      language=self.cleaned_data['language'],
      image_size=self.cleaned_data['image_size'],
      visualize=self.cleaned_data['visualize'],
      status='queued',
    )
    run_ocr_job.enqueue(job.pk)
    return job
