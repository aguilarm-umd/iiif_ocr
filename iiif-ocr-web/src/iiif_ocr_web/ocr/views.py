from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render

from .forms import ManifestSubmissionForm, OCRJobForm
from .models import Manifest
from .tasks import run_ocr_job


def home(request):
  if request.method == 'POST':
    form = ManifestSubmissionForm(request.POST)
    if form.is_valid():
      job = form.save()
      messages.success(request, f'OCR job {job.number} was added.')
      return redirect('manifest-detail', manifest_id=job.manifest_id)
  else:
    form = ManifestSubmissionForm()

  manifests = Manifest.objects.prefetch_related('jobs').order_by('-created_at')
  return render(request, 'ocr/home.html', {'form': form, 'manifests': manifests})


def manifest_detail(request, manifest_id):
  manifest = get_object_or_404(Manifest.objects.prefetch_related('jobs'), id=manifest_id)
  if request.method == 'POST':
    form = OCRJobForm(request.POST)
    if form.is_valid():
      job = form.save(commit=False)

      job.manifest = manifest
      job.status = 'queued'
      job.save()

      run_ocr_job.enqueue(job.pk)
      messages.success(request, f'OCR job {job.number} was added.')

      return redirect('manifest-detail', manifest_id=manifest.id)
  else:
    form = OCRJobForm()

  return render(request, 'ocr/manifest_detail.html', {'manifest': manifest, 'form': form})
