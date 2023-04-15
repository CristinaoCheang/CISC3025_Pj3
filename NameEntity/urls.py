
from django.urls import path
from . import views


app_name = "NameEntity"
urlpatterns = [
    path('', views.classify_sentence, name='classify_sentence'),
]
