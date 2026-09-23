from django.urls import path

from rag import views

urlpatterns = [
    path("", views.ask, name="ask"),
]
