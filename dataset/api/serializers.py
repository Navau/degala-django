from rest_framework.serializers import ModelSerializer

from dataset.models import DataSet


class DataSetSerializer(ModelSerializer):
    class Meta:
        model = DataSet
        fields = ["id", "date", "sales", "quantity"]
