from django.db import models

# Create your models here.
from rest_framework import serializers

class VideoProcessingSerializer(serializers.Serializer):
    source_video = serializers.FileField()
    username = serializers.CharField(max_length=100)
