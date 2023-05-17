from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('', view = views.home, name='home'),
    path('imageprocess', view= views.imageprocess, name= 'imageprocess')
]

