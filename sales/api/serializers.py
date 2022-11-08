from rest_framework.serializers import ModelSerializer

from sales.models import Sale

from users.api.serializers import UserSerializer


class SaleSerializer(ModelSerializer):
    user_data = UserSerializer(source='user', read_only=True)

    class Meta:
        model = Sale
        fields = ["id", "totalSale", "totalQuantity",
                  "totalDiscount", "created_at", "statusSale", "user", "user_data"]
