import React from 'react';
import './ContactUs.css'; // Import the CSS file

const ContactUs = () => {
    return (
        <div className="contact-us-container">
            <h2>Contact Us Page</h2>
            <form className="contact-form">
                <input type="text" placeholder="Your Name" required />
                <input type="email" placeholder="Your Email" required />
                <textarea placeholder="Your Message" rows="4" required></textarea>
                <button type="submit">Send Message</button>
            </form>
        </div>
    );
};

export default ContactUs;
