from django.db import models


class SaleDetail(models.Model):
    sale = models.ForeignKey(
        'sales.Sale', on_delete=models.SET_NULL, null=True, blank=True)
    product = models.ForeignKey(
        'products.Product', on_delete=models.SET_NULL, null=True, blank=True)
    category = models.ForeignKey(
        'categories.Category', on_delete=models.SET_NULL, null=True, blank=True)
    quantity = models.IntegerField(default=0)
    payment = models.DecimalField(max_digits=5, decimal_places=2)
    comment = models.CharField(max_length=1000, null=True, blank=True)
