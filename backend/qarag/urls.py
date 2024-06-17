from django.urls import path
from . import views

urlpatterns = [
    path('', views.qa_rag, name='qa_rag'),
    path('form/', views.qa_form, name='qa_form'),
]
