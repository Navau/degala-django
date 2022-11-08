from rest_framework.serializers import ModelSerializer

from salesDetails.models import SaleDetail


class SaleDetailSerializer(ModelSerializer):
    class Meta:
        model = SaleDetail
        fields = ["id", "sale", "product", "category",
                  "quantity", "payment", "comment"]
