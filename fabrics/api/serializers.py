from rest_framework.serializers import ModelSerializer

from fabrics.models import Fabric


class FabricSerializer(ModelSerializer):
    class Meta:
        model = Fabric
        fields = ['id', 'title', 'price', 'description', 'active']
