import React, { useState, useEffect } from 'react';
import './Notworking.css';

const Notworking = () => {
    const [capturedImage, setCapturedImage] = useState(null);
    const [detectionDetails, setDetectionDetails] = useState(null);
    const [classificationResult, setClassificationResult] = useState('No phone detected.');

    // Function to capture the image
    const handleCapture = () => {
        fetch('/capture_image', { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.image_url) {
                    setCapturedImage(data.image_url);
                }
                setDetectionDetails(data.details);
                setClassificationResult(data.details.classification || 'No phone detected.');
            })
            .catch(error => console.error('Error:', error));
    };

    return (
        <div className="notworking-container">
            <h1>Object Detection and Classification</h1>

            {/* Video feed */}
            <div className="video-container">
                <img src="/video_feed" alt="Video Feed" className="video-feed" />
            </div>

            {/* Button to capture image */}
            <button onClick={handleCapture} className="capture-button">Capture Image</button>

            {/* Display captured image */}
            <div className="captured-image-container">
                <h3>Captured Image:</h3>
                {capturedImage && <img src={capturedImage} alt="Captured" className="captured-image" />}
            </div>

            {/* Display detection details */}
            <div className="details-container">
                <h3>Detection Details:</h3>
                {detectionDetails && <pre>{JSON.stringify(detectionDetails, null, 2)}</pre>}
            </div>

            {/* Display classification results */}
            <div className="classification-container">
                <h3>Classification Results:</h3>
                <p>{classificationResult}</p>
            </div>
        </div>
    );
};

export default Notworking;
