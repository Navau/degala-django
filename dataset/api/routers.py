from django.urls import path
from rest_framework.routers import DefaultRouter

from dataset.api.views import DataSetViewSet, UploadExcel

router_dataset = DefaultRouter()
router_dataset.register(prefix="dataset", basename="dataset", viewset=DataSetViewSet)

urlpatterns = [
    # path("upload_excel/", UploadExcel.as_view(), name="upload_excel"),
]
