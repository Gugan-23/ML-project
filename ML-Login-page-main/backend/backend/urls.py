from django.contrib import admin
from django.urls import path
from django.http import HttpResponse
from accounts.views import SignupView, LoginView

from .views import upload_appliance,list_uploaded_files,user_details
# Simple view for root URL
def home(request):
    return HttpResponse('<h1>Welcome to the Django-Backend..!'
                    '<div>Your Django Server connected Successfully..!</h1> </div> ')

urlpatterns = [
    path('', home, name='home'),  # Root URL now points to this view
    path('signup/', SignupView.as_view(), name='signup'),
    path('login/', LoginView.as_view(), name='login'),
    path('userdetails/', user_details, name='user_details'),
    path('upload/', upload_appliance, name='upload-appliance'),
    path('uploaded-files/', list_uploaded_files, name='list_uploaded_files'),  # New URL for uploaded files

]
