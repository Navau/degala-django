from django.db import models


class Fabric(models.Model):
    title = models.CharField(max_length=255)
    price = models.DecimalField(max_digits=6, decimal_places=2)
    description = models.CharField(max_length=1000)
    active = models.BooleanField(default=False)

    def __str__(self):
        return self.title
