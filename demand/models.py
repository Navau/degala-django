from django.db import models


class Demand(models.Model):
    date = models.CharField(max_length=255, null=True)
    sales = models.IntegerField(default=-1)
    quantity = models.IntegerField(default=-1)

    def __str__(self):
        return self.date
