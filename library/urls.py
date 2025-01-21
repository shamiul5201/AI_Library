from django.contrib import admin
from django.urls import path, include
from core import views
from face_applications import views

urlpatterns = [
    path("admin/", admin.site.urls),
    path('', include('core.urls')),
    path('', include('face_applications.urls')),

]

