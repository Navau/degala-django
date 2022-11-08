from django.contrib import admin
from salesDetails.models import SaleDetail


@admin.register(SaleDetail)
class SaleDetailAdmin(admin.ModelAdmin):
    pass
