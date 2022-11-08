from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from django_filters.rest_framework import DjangoFilterBackend

from sales.models import Sale
from sales.api.serializers import SaleSerializer


class SaleApiViewSet(ModelViewSet):
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = SaleSerializer
    queryset = Sale.objects.all()
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["totalSale", "totalQuantity",
                        "totalDiscount", "user", "statusSale"]
