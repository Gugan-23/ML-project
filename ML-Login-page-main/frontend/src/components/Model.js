import React, { useState } from 'react';

const Model = () => {
    const [formData, setFormData] = useState({
        brand: ''
    });

    const [predictions, setPredictions] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Handle input changes
    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    // Handle form submission
    // Handle form submission
    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        
        try {
            const response = await fetch(`http://127.0.0.1:5000/predict`, {
                method: 'POST',  // Specify POST method
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ brand: formData.brand }),  // Send brand in the body
            });
    
            if (!response.ok) {
                throw new Error('Failed to fetch the prediction. Please try again.');
            }
    
            const data = await response.json();
            setPredictions(data);
            setFormData({ brand: '' });  // Clear input after submission
        } catch (error) {
            setError(error.message);  // Set error state
        } finally {
            setLoading(false);  // Reset loading state
        }
    };
    
    

    return (
        <div>
            <h2>Predict the Element Quantities</h2>
            <form onSubmit={handleSubmit}>
                <input
                    type="text"
                    name="brand"
                    placeholder="Enter Brand"
                    value={formData.brand}
                    onChange={handleChange}
                    required
                />
                <button
                    type="submit"
                    disabled={loading}
                >
                    {loading ? 'Predicting...' : 'Predict Quantities'}
                </button>
            </form>
            {error && (
                <div aria-live="assertive">
                    <p>{error}</p>
                </div>
            )}
            {predictions && !error && (
                <div aria-live="polite">
                    <h3>Estimated Quantities:</h3>
                    <ul>
                        <li>Gold: {predictions['Gold (g)'].toFixed(2)} g</li>
                        <li>Aluminum: {predictions['Aluminum (g)'].toFixed(2)} g</li>
                        <li>Silver: {predictions['Silver (g)'].toFixed(2)} g</li>
                        <li>Carbon: {predictions['Carbon (g)'].toFixed(2)} g</li>
                        <li>Platinum: {predictions['Platinum (g)'].toFixed(2)} g</li>
                        <li>Nickel: {predictions['Nickel (g)'].toFixed(2)} g</li>
                        <li>Lithium: {predictions['Lithium (g)'].toFixed(2)} g</li>
                        <li>Price :{predictions['Estimated Price (INR)'].toFixed(2)}</li>
                    </ul>
                </div>
            )}
        </div>
    );
};

export default Model;
