from . import views, viewsRecords
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),  # 根路径直接访问上传页面
    path('calculate/', views.calculate, name='calculate'),
    path('home/', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('record_list/', viewsRecords.record_list, name='record_list'),
    path('upload_file/', viewsRecords.upload_file, name='upload_file'),
    path('download/<int:record_id>/<str:file_type>/', viewsRecords.download_file, name='download_file'),
    path('illustration/', views.illustration, name='illustration'),
    path('ML/', views.ML, name='ML'),
    path("predict/", views.predict, name="predict"),
    path('upload/', views.upload_excel, name='upload_excel'),
    path('process/', views.process_excel, name='process_excel'),
    path('feature_engineering/', views.feature_engineering, name='feature_engineering'),
]
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
