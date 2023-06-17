from rest_framework.serializers import ModelSerializer

from demand.models import Demand


class DemandSerializer(ModelSerializer):
    class Meta:
        model = Demand
        fields = ["date", "sales", "quantity"]
