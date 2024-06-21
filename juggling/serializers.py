from rest_framework import serializers
from .models import VideoProcessingSerializer

class VideoSerializer(serializers.ModelSerializer):
    class Meta:
        model = VideoProcessingSerializer
        fields = '__all__'