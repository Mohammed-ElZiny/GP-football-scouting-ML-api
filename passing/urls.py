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
    path('passing/', process_video_api, name='passing_api'),
] 
