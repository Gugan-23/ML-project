// AboutUs.js

import React from 'react';
import './AboutUs.css'; // Import the CSS file
import MovableGraph from './MovableGraph'; // Import the MovableGraph component

const AboutUs = () => {
    return (
        <div className="about-us-container">
            <h1>About Us</h1>
            <p>We are dedicated to promoting environmental awareness through e-waste management.</p>
            <p>Our mission is to educate the public on the importance of recycling e-waste responsibly.</p>

            {/* Add the MovableGraph component here */}
            <div className="movable-graph">
                <MovableGraph />
            </div>
        </div>
    );
};

export default AboutUs;
