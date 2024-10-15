import React from 'react';
import { useNavigate } from 'react-router-dom';

const Profile = () => {
    const navigate = useNavigate();
    const user = JSON.parse(localStorage.getItem('user')); // Retrieve user data from local storage

    const handleLogout = () => {
        localStorage.removeItem('user'); // Clear user data from local storage
        navigate('/accounts/login'); // Redirect to login page
    };

    if (!user) {
        return (
            <div>
                <h2>Please log in to view your profile.</h2>
                <button onClick={() => navigate('/login')}>Login</button>
            </div>
        );
    }

    return (
        <div className="profile-container">
            <h2>User Profile</h2>
            <p><strong>Email:</strong> {user.email}</p> {/* Adjust based on your user object structure */}
            <p><strong>Name:</strong> {user.name}</p> {/* Assuming there's a name field */}
            {/* Add more fields as necessary */}
            <button onClick={handleLogout}>Logout</button>
        </div>
    );
};

export default Profile;
