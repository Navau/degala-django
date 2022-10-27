from rest_framework.routers import DefaultRouter
from fabrics.api.views import FabricApiViewSet

router_fabric = DefaultRouter()

router_fabric.register(
    prefix='fabrics', basename='fabrics', viewset=FabricApiViewSet
)
