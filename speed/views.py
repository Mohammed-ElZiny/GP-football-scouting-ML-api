from django.shortcuts import render

# Create your views here.
import os
from datetime import datetime
from django.conf import settings
from django.core.files.temp import NamedTemporaryFile
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status

from speed.video_processing.video_processor import process_video

# from video_processing.video_processor import process_video
from .serializers import VideoProcessingSerializer

@api_view(['POST'])
def process_video_api(request):
    if request.method == 'POST':
        serializer = VideoProcessingSerializer(data=request.data)
        if serializer.is_valid():
            source_video = serializer.validated_data['source_video']
            username = serializer.validated_data['username']
            
            # Create a temporary file to store the uploaded video
            with NamedTemporaryFile() as temp_file:
                # Write the uploaded file content to the temporary file
                for chunk in source_video.chunks():
                    temp_file.write(chunk)
                temp_file.flush()
                
                # Construct the target video filename with username and date
                date_today = datetime.now().strftime("%Y%m%d")
                target_video_filename = f"{username}_speed_{date_today}.mp4"   # or any other desired video extension

                # Construct the target video path with username and date
                target_video_dir = os.path.join(settings.MEDIA_ROOT, 'userdata', username)
                os.makedirs(target_video_dir, exist_ok=True)
                target_video_path = os.path.join(target_video_dir, target_video_filename)

                try:
                    # Process the video and store it in the target path
                    average_speed = process_video(temp_file.name, target_video_path)
                    
                    # Generate URL for the target video path
                    target_video_url = os.path.join(settings.MEDIA_URL, 'userdata', username, target_video_filename)
                    
                    # Return response with username, average_speed, and target video path
                    return Response({'username': username, 'average_speed': average_speed, 'target_video_path': target_video_url}, status=status.HTTP_200_OK)
                except Exception as e:
                    return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
