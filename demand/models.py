from django.db import models


class Demand(models.Model):
    mes = models.CharField(max_length=255)
    ventas = models.IntegerField()
    pred_value = models.IntegerField()
    real_value = models.IntegerField()

    def __str__(self):
        return self.pred_value
