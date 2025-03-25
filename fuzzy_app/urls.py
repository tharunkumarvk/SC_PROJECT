from django.urls import path
from . import views

app_name = 'fuzzy_app'
urlpatterns = [
    path('', views.index, name='index'),
    path('about/', views.about, name='about'),
    path('initdb/', views.init_db, name='init_db'),  # For development only
]