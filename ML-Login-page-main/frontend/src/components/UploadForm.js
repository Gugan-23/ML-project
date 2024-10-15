import React, { useState } from 'react';

const UploadForm = () => {
    const [applianceType, setApplianceType] = useState('');
    const [details, setDetails] = useState('');
    const [photo, setPhoto] = useState(null);
    const [message, setMessage] = useState('');
    const [uploadedData, setUploadedData] = useState(null); // New state to hold uploaded data

    const handleSubmit = async (e) => {
        e.preventDefault();

        const formData = new FormData();
        formData.append('appliance_type', applianceType); // Match the field name in Django model
        formData.append('details', details);
        formData.append('photo', photo);

        try {
            const response = await fetch('http://localhost:8000/upload/', { // Use your Django server URL
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const jsonResponse = await response.json();
                setMessage(jsonResponse.message);
                
                // Set the uploaded data for display
                setUploadedData({
                    applianceType: applianceType,
                    details: details,
                    photoUrl: jsonResponse.file_url // Assuming your Django API returns the file URL
                });

                // Reset form fields
                setApplianceType('');
                setDetails('');
                setPhoto(null);
            } else {
                const errorResponse = await response.json();
                setMessage('Upload failed: ' + errorResponse.detail);
            }
        } catch (error) {
            console.error('Error:', error);
            setMessage('An error occurred. Please try again.');
        }
    };

    return (
        <div>
            <form onSubmit={handleSubmit}>
                <h2>Upload Appliance or Electronic Details</h2>
                <div>
                    <label>Appliance Type:</label>
                    <input
                        type="text"
                        value={applianceType}
                        onChange={(e) => setApplianceType(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label>Details:</label>
                    <textarea
                        value={details}
                        onChange={(e) => setDetails(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label>Photo:</label>
                    <input
                        type="file"
                        accept="image/*"
                        onChange={(e) => setPhoto(e.target.files[0])}
                        required
                    />
                </div>
                <button type="submit">Upload</button>
                {message && <p>{message}</p>} {/* Display success or error message */}
            </form>

            {/* Show uploaded content if available */}
            {uploadedData && (
                <div>
                    <h3>Uploaded Details</h3>
                    <p><strong>Appliance Type:</strong> {uploadedData.applianceType}</p>
                    <p><strong>Details:</strong> {uploadedData.details}</p>
                    {uploadedData.photoUrl && (
                        <div>
                            <strong>Photo:</strong>
                            <img src={uploadedData.photoUrl} alt="Uploaded Appliance" style={{ maxWidth: '200px', maxHeight: '200px' }} />
                        </div>
                    )}
                </div>
            )}
        </div>
    );
};

export default UploadForm;
