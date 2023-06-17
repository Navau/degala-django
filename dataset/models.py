from django.db import models


class DataSet(models.Model):
    date = models.CharField(max_length=20, null=True)
    sales = models.CharField(max_length=20, null=True)
    quantity = models.CharField(max_length=20, null=True)
