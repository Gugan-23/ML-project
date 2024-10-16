import React from 'react';

const Predictip = () => {
    const handleButtonClick = async () => {
        try {
            const response = await fetch('http://localhost:5000/open-window', {
                method: 'GET',
            });
            const message = await response.text();
            console.log(message); // Optional: Log the response from the server
        } catch (error) {
            console.error('Error opening window:', error);
        }
    };

    return (
        <div>
            <h2>Hi</h2>
            <button onClick={handleButtonClick}>Open Tkinter Window</button>
        </div>
    );
};

export default Predictip;
