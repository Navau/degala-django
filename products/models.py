from django.db import models

GenreProductEnum = (
    ("MALE", "male"),
    ("FEMALE", "female"),
    ("UNDEFINED", "undefined"),
    ("OTHER", "other"),
)


class Product(models.Model):
    title = models.CharField(max_length=255)
    image = models.ImageField(upload_to='products')
    price = models.DecimalField(max_digits=7, decimal_places=2)  # 99999.99
    color = models.CharField(max_length=255, null=True, blank=True)
    stock = models.IntegerField(default=0)
    genre = models.CharField(
        default="undefined", max_length=255, choices=GenreProductEnum)
    description = models.CharField(default="", max_length=1000)
    active = models.BooleanField(default=False)
    fabric = models.ForeignKey(
        'fabrics.Fabric', on_delete=models.SET_NULL, null=True, blank=True)
    category = models.ForeignKey(
        'categories.Category', on_delete=models.SET_NULL, null=True, blank=True)

    def __str__(self):
        return self.title
