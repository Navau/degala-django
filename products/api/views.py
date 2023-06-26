from rest_framework.viewsets import ModelViewSet
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.filters import SearchFilter

from products.models import Product
from products.api.serializers import ProductSerializer


class ProductApiViewSet(ModelViewSet):
    permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = ProductSerializer
    queryset = Product.objects.all()
    filter_backends = [DjangoFilterBackend, SearchFilter]
    filterset_fields = ["category", "active"]
    search_fields = [
        "title",
        "image",
        "price",
        "color",
        "stock",
        "genre",
        "description",
        "fabric__title",
        "category__title",
    ]
