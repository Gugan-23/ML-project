from django.shortcuts import render

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import authenticate
from .models import MyUser
from django.contrib.auth.models import update_last_login
from rest_framework.authtoken.models import Token
from django.contrib.auth import get_user_model
from django.db import IntegrityError

User = get_user_model()

from rest_framework import generics
from .serializers import UserSerializer  # Create this serializer

class SignupView(generics.CreateAPIView):
    queryset = MyUser.objects.all()
    serializer_class = UserSerializer  # Use a serializer to validate data

    def create(self, request, *args, **kwargs):
        username = request.data.get('username')
        email = request.data.get('email')
        password = request.data.get('password')
        confirm_password = request.data.get('confirm_password')
        phone_number = request.data.get('phone_number')

        if password != confirm_password:
            return Response({"detail": "Passwords do not match."}, status=status.HTTP_400_BAD_REQUEST)

        user = MyUser(username=username, email=email, phone_number=phone_number)
        user.set_password(password)
        user.save()

        return Response({"detail": "Signup successful!"}, status=status.HTTP_201_CREATED)




class LoginView(APIView):
    def post(self, request):
        email = request.data.get('email')
        password = request.data.get('password')

        user = authenticate(request, email=email, password=password)

        if user is not None:
            token, _ = Token.objects.get_or_create(user=user)
            return Response({'token': str(token)}, status=status.HTTP_200_OK)
        else:
            return Response({"detail": "Invalid credentials"}, status=status.HTTP_400_BAD_REQUEST)
