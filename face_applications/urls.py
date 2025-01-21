from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('face_alignment/', views.face_alignment_view, name='face_alignment'),
    # path('delaunay_triangle/', views.upload_image, name='delaunay_triangle')
] 

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)