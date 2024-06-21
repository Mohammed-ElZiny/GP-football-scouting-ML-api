from django.db import models

# Create your models here.
# # myapp/serializers.py

# from rest_framework import serializers
# from django.core.files.uploadedfile import InMemoryUploadedFile

# class VideoProcessingSerializer(serializers.Serializer):
#     source_video = serializers.FileField()
from rest_framework import serializers

class VideoProcessingSerializer(serializers.Serializer):
    source_video = serializers.FileField()
    username = serializers.CharField(max_length=100)
