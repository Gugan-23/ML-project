import React, { useState } from 'react';

const Predictip = () => {
    // State to hold input values
    const [brand, setBrand] = useState('');
    const [name, setName] = useState('');
    const [displaySize, setDisplaySize] = useState('');
    const [specifications, setSpecifications] = useState(null); // State to hold the updated specifications
    const [productType, setProductType] = useState(''); // State to hold the predicted product type

    // Handle form submission
    const handleFormSubmit = async (event) => {
        event.preventDefault(); // Prevent the default form submission behavior

        try {
            // Send a POST request to the Flask backend
            const response = await fetch('http://localhost:5000/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    brand,
                    name,
                    display_size: displaySize, // Send display size as a parameter
                }),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            // Parse the JSON response
            const product_type = await response.json();
            setSpecifications(product_type); // Update the state with the response data
            setProductType(product_type); // Update the state with the predicted product type
            console.log('Updated specifications:', product_type); // Log the updated specifications
        } catch (error) {
            console.error('Error submitting form:', error);
        }
    };

    return (
        <div>
            <h2>Submit Product Specification</h2>
            <form onSubmit={handleFormSubmit}>
                <div>
                    <label htmlFor="brand">Brand:</label>
                    <input
                        type="text"
                        id="brand"
                        value={brand}
                        onChange={(e) => setBrand(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label htmlFor="name">Product Name:</label>
                    <input
                        type="text"
                        id="name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label htmlFor="display_size">Display Size:</label>
                    <input
                        type="number"
                        id="display_size"
                        value={displaySize}
                        onChange={(e) => setDisplaySize(e.target.value)}
                        required
                    />
                </div>
                <button type="submit">Submit</button>
            </form>
            
            {productType && (
                <div>
                    <h3>Predicted Product Type:</h3>
                    <p>{productType}</p>
                </div>
            )}
        </div>
    );
};

export default Predictip;
