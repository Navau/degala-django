from django.contrib import admin
from demand.models import Demand


@admin.register(Demand)
class DemandAdmin(admin.ModelAdmin):
    pass
