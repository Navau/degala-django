from rest_framework.routers import DefaultRouter
from sales2.api.views import Sale2ApiViewSet

router_sale2 = DefaultRouter()

router_sale2.register(
    prefix="sales2", basename="sales2", viewset=Sale2ApiViewSet
)
