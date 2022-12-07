from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from django_filters.rest_framework import DjangoFilterBackend

from sales2.models import Sale2
from sales2.api.serializers import Sale2Serializer


class Sale2ApiViewSet(ModelViewSet):
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = Sale2Serializer
    queryset = Sale2.objects.all()
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ["product", "category", "quantity", "active"]
