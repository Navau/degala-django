from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from django_filters.rest_framework import DjangoFilterBackend

from fabrics.models import Fabric
from fabrics.api.serializers import FabricSerializer


class FabricApiViewSet(ModelViewSet):
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = FabricSerializer
    queryset = Fabric.objects.all()
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["active"]
