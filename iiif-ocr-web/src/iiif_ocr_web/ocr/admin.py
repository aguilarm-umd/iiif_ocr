from django.contrib import admin

from .models import Manifest, OCRJob, Page


@admin.register(Manifest)
class ManifestAdmin(admin.ModelAdmin):
  list_display = ('id', 'url', 'status', 'created_at')
  search_fields = ('url', 'id')


@admin.register(OCRJob)
class OCRJobAdmin(admin.ModelAdmin):
  list_display = ('number', 'manifest', 'status', 'current_page', 'created_at')
  list_filter = ('status', 'language', 'image_size')


@admin.register(Page)
class PageAdmin(admin.ModelAdmin):
  list_display = ('job', 'page_number', 'status')
  list_filter = ('status',)
