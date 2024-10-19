import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom'; // Import useNavigate
import './Profile.css';
const Profile = () => {
    const [profile, setProfile] = useState({
        username: '',
        email: '',
        phone_number: ''
    });

    const token = localStorage.getItem('authToken'); // Get the token stored in localStorage
    const navigate = useNavigate(); // Initialize useNavigate

    useEffect(() => {
        // Fetch user profile details
        const fetchProfile = async () => {
            if (!token) {
                console.error("No token found. Please log in.");
                return; // Exit if there is no token
            }

            try {
                const response = await axios.get('http://127.0.0.1:8000/profile/', {
                    headers: {
                        'Authorization': `Token ${token}` // Include the token in the Authorization header
                    }
                });
                setProfile(response.data); // Set the fetched profile data
            } catch (err) {
                console.error("Error fetching profile:", err);
            }
        };

        fetchProfile();
    }, [token]); // Fetch profile whenever the token changes

    // Logout function
    const handleLogout = () => {
        localStorage.removeItem('authToken'); // Remove the token from localStorage
        navigate('/login'); // Redirect to login page
    };

    return (
        <div>
            <h1>My Profile</h1>
            <div>
                <label><strong>Name:</strong></label>
                <p>{profile.username}</p>
            </div>
            <div>
                <label><strong>Email:</strong></label>
                <p>{profile.email}</p>
            </div>
            <div>
                <label><strong>Phone:</strong></label>
                <p>{profile.phone_number}</p>
            </div>
            <button onClick={handleLogout}>Logout</button> {/* Logout button */}
        </div>
    );
};

export default Profile;
