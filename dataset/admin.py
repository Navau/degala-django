from django.contrib import admin
from dataset.models import DataSet


@admin.register(DataSet)
class DataSetAdmin(admin.ModelAdmin):
    pass
