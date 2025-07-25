from django.urls import path 
from . import views 

app_name = "ml_app"

urlpatterns = [
    path("", views.index, name="home"),
    path("predict/", views.predict_page, name="predict"),
    path("about/", views.about, name="about"),
    path("cuda_full/", views.cuda_full, name="cuda_full"),
]

#Custom 404 handler (custom 404 page)
handler404 = 'ml_app.views.handler404'
