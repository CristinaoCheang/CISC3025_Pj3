from django.urls import path
from .views import home


app_name = "NameEntity"
urlpatterns = [
    path("", home, name="home"),
]
