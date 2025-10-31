from django.db import models


# Create your models here.
class Record(models.Model):
    structure_file = models.FileField(upload_to='sturestrus/')
    original_structure_name = models.CharField(max_length=255)  # 存储原始结构文件名
    features_file = models.FileField(upload_to='features/')
    original_features_name = models.CharField(max_length=255)  # 存储原始特征文件名
    performance_k = models.CharField(max_length=255)
    performance_tan_omega = models.CharField(max_length=255)
    performance_tck = models.CharField(max_length=255)
    note = models.TextField(blank=True)
    reference = models.CharField(max_length=255)
    uploader = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)




