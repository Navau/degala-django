from rest_framework.serializers import ModelSerializer

from products.models import Product

from categories.api.serializers import CategorySerializer
from fabrics.api.serializers import FabricSerializer


class ProductSerializer(ModelSerializer):
    category_data = CategorySerializer(source='category', read_only=True)
    fabric_data = FabricSerializer(source='fabric', read_only=True)

    class Meta:
        model = Product
        fields = ['id', 'title', 'image', 'price', 'color', 'stock', 'genre', 'description',
                  'active', 'fabric', 'fabric_data', 'category', 'category_data']
