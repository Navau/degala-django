from rest_framework.viewsets import ModelViewSet
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticatedOrReadOnly
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework import status
from rest_framework.response import Response
import pandas as pd
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.decorators import action


from dataset.models import DataSet
from dataset.api.serializers import DataSetSerializer


class DataSetViewSet(ModelViewSet):
    # permission_classes = [IsAuthenticatedOrReadOnly]
    serializer_class = DataSetSerializer
    queryset = DataSet.objects.all()
    filter_backends = [DjangoFilterBackend]
    filterset_fields = {"date": ["range"], "sales": ["exact"], "quantity": ["exact"]}


class UploadExcel(APIView):
    def post(self, request):
        file = request.FILES.get("excel_file")
        if file:
            df = pd.read_csv(file)
            df["quantity"] = 1
            df_product_count = (
                df.groupby(["date", "product"])["price", "quantity"].sum().reset_index()
            )
            df = (
                df_product_count.groupby("date")["price", "quantity"]
                .sum()
                .reset_index()
            )
            for _, row in df.iterrows():
                date = row["date"]
                sales = row["price"]
                quantity = row["quantity"]

                DataSet.objects.create(date=date, sales=sales, quantity=quantity)

            return Response(
                {"message": "Archivo Excel subido y registrado correctamente."},
                status=status.HTTP_200_OK,
            )
        else:
            return Response(
                {"error": "No se ha proporcionado ningún archivo Excel."},
                status=status.HTTP_400_BAD_REQUEST,
            )
