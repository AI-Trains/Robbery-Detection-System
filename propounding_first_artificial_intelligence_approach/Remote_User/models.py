from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class robbery_behavior_detection(models.Model):

    Fid= models.CharField(max_length=300)
    event_unique_id= models.CharField(max_length=300)
    occurrencedate= models.CharField(max_length=300)
    reporteddate= models.CharField(max_length=300)
    location_type= models.CharField(max_length=300)
    premises_type= models.CharField(max_length=300)
    Neighbourhood= models.CharField(max_length=300)
    Longitude= models.CharField(max_length=300)
    Latitude= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)


class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



