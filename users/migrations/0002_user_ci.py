# Generated by Django 4.1.2 on 2022-11-06 22:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('users', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='user',
            name='ci',
            field=models.CharField(blank=True, max_length=100, null=True),
        ),
    ]
