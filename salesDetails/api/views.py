from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from django_filters.rest_framework import DjangoFilterBackend

from salesDetails.models import SaleDetail
from salesDetails.api.serializers import SaleDetailSerializer


class SaleDetailApiViewSet(ModelViewSet):
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = SaleDetailSerializer
    queryset = SaleDetail.objects.all()
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["sale", "product",
                        "category", "quantity", "payment"]
