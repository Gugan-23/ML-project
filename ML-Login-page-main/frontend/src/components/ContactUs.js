import React, { useState } from 'react';
import axios from 'axios';
import './ContactUs.css';
function ContactUs() {
  const [formData, setFormData] = useState({ name: '', email: '', message: '' });
  const [statusMessage] = useState('');

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    axios.post('https://personalportfolio-api.vercel.app/', formData)
      .then(response => {
        
        alert('Email sent successfully!'); // Display dialog box
      })
      .catch(error => {
        
        console.error('There was an error sending the email!', error);
        alert('Failed to send email. Please try again later.'); // Display dialog box
      });
  };

  return (
    <section id="contact">
      <h2>Contact Me</h2>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          name="name"
          placeholder="Your Name"
          value={formData.name}
          onChange={handleChange}
          required
        />
        <input
          type="email"
          name="email"
          placeholder="Your Email"
          value={formData.email}
          onChange={handleChange}
          required
        />
        <textarea
          name="message"
          placeholder="Your Message"
          value={formData.message}
          onChange={handleChange}
          required
        ></textarea>
        <button type="submit">Submit</button>
      </form>
      {statusMessage && <p>{statusMessage}</p>}
    </section>
  );
}

export default ContactUs;