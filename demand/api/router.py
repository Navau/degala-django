from django.urls import path
from rest_framework.routers import DefaultRouter

from demand.api.views import (
    DemandApiViewSet,
    PredictMonthView,
    PredictView,
    DemandPredictMonthView,
)

router_demand = DefaultRouter()

router_demand.register(prefix="demand", basename="demand", viewset=DemandApiViewSet)

urlpatterns = [
    path("predict/", PredictView.as_view(), name="predict_view"),
    path("predict/<int:month>/", PredictMonthView.as_view(), name="predict_month_view"),
    path(
        "demand/predict-month/",
        DemandPredictMonthView.as_view(),
        name="demand_predict_month",
    ),
]
