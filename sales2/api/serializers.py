from rest_framework.serializers import ModelSerializer

from sales2.models import Sale2

from users.api.serializers import UserSerializer
from products.api.serializers import ProductSerializer
from categories.api.serializers import CategorySerializer


class Sale2Serializer(ModelSerializer):
    user_data = UserSerializer(source='user', read_only=True)
    product_data = ProductSerializer(source='product', read_only=True)
    category_data = CategorySerializer(source='category', read_only=True)

    class Meta:
        model = Sale2
        fields = ["id", "quantity", "payment", "change", "comment", "created_at", "active",
                  "user", "user_data", "product", "product_data", "category", "category_data"]
