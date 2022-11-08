from rest_framework.routers import DefaultRouter
from salesDetails.api.views import SaleDetailApiViewSet

router_saleDetail = DefaultRouter()

router_saleDetail.register(
    prefix="salesDetails", basename="salesDetails", viewset=SaleDetailApiViewSet
)
