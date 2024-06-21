# from django.urls import path
# from .views import process_video_api 


# urlpatterns = [
#     path('process-video/', process_video_api, name='process-video'),
   
# ]
# urls.py
from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from .views import process_video_api

urlpatterns = [
    path('zigzag/', process_video_api, name='zigzag_api'),
]
