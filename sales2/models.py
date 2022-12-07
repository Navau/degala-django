from django.db import models


class Sale2(models.Model):
    product = models.ForeignKey(
        'products.Product', on_delete=models.SET_NULL, null=True, blank=True)
    category = models.ForeignKey(
        'categories.Category', on_delete=models.SET_NULL, null=True, blank=True)
    user = models.ForeignKey(
        "users.User", on_delete=models.SET_NULL, null=True, blank=True)
    quantity = models.IntegerField(default=0)
    payment = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    change = models.DecimalField(max_digits=5, decimal_places=2, default=0)
    comment = models.CharField(max_length=1000, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    active = models.BooleanField(default=True)
