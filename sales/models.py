from django.db import models


class Sale(models.Model):
    totalSale = models.DecimalField(
        max_digits=16, decimal_places=2)  # 99999999999999.99
    totalQuantity = models.IntegerField(default=0)
    totalDiscount = models.DecimalField(max_digits=16, decimal_places=2)
    totalPayment = models.DecimalField(max_digits=16, decimal_places=2)
    created_at = models.DateTimeField(auto_now_add=True)
    statusSale = models.BooleanField(default=True)
    user = models.ForeignKey(
        "users.User", on_delete=models.SET_NULL, null=True, blank=True)
