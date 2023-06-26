from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter

from sales2.models import Sale2
from sales2.api.serializers import Sale2Serializer


class Sale2ApiViewSet(ModelViewSet):
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = Sale2Serializer
    queryset = Sale2.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter]
    filterset_fields = ["product", "category", "quantity", "active"]
    search_fields = [
        "quantity",
        "payment",
        "change",
        "comment",
        "created_at",
        "user__username",
        "user__first_name",
        "user__last_name",
        "product__title",
        "category__title",
    ]
