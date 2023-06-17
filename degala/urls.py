"""degala URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include

from django.conf import settings
from django.conf.urls.static import static

# from rest_framework import permissions
from drf_yasg.views import get_schema_view
from drf_yasg import openapi

from users.api.router import router_user
from demand.api.router import router_demand
from categories.api.router import router_category
from products.api.router import router_product
from fabrics.api.router import router_fabric
from sales.api.router import router_sale
from salesDetails.api.router import router_saleDetail
from sales2.api.router import router_sale2
from dataset.api.routers import router_dataset

schema_view = get_schema_view(
    openapi.Info(
        title="DeGala API DOCS",
        default_version="v1",
        description="Documentacion de la api degala",
        terms_of_service="https://www.google.com/policies/terms/",
        contact=openapi.Contact(email="contactojosegutierrez@gmail.com"),
        license=openapi.License(name="BSD License"),
    ),
    public=True,
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path(
        "docs/",
        schema_view.with_ui("swagger", cache_timeout=0),
        name="schema_swagger_ui",
    ),
    path("redocs/", schema_view.with_ui("redoc", cache_timeout=0), name="schema-redoc"),
    path("api/", include("users.api.router")),
    path("api/", include("demand.api.router")),
    path("api/", include("dataset.api.routers")),
    path("api/", include(router_user.urls)),
    path("api/", include(router_demand.urls)),
    path("api/", include(router_category.urls)),
    path("api/", include(router_product.urls)),
    path("api/", include(router_fabric.urls)),
    path("api/", include(router_sale.urls)),
    path("api/", include(router_saleDetail.urls)),
    path("api/", include(router_sale2.urls)),
    path("api/", include(router_dataset.urls)),
]
# ] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
