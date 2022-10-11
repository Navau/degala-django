from distutils.command.upload import upload
from email.mime import image
from unicodedata import category
from django.db import models


class Product(models.Model):
    title = models.CharField(max_length=255)
    image = models.ImageField(upload_to='´/products')
    price = models.DecimalField(max_digits=5, decimal_places=2)  # 999.99
    active = models.BooleanField(default=False)
    category = models.ForeignKey(
        'categories.Category', on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return self.title
