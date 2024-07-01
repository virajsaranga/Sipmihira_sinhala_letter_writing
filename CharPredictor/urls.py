from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('request-video/', request_video),
    path('submit-char/', submit_char),
]
