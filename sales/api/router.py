from rest_framework.routers import DefaultRouter
from sales.api.views import SaleApiViewSet

router_sale = DefaultRouter()

router_sale.register(
    prefix="sales", basename="sales", viewset=SaleApiViewSet
)
