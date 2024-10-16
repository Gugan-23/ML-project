import React, { useEffect, useRef, useState } from 'react';
import './Notworking.css'; // Adjust the path if necessary

const Notworking = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [isCameraOn, setIsCameraOn] = useState(false);
    const [photoUrl, setPhotoUrl] = useState(null);

    useEffect(() => {
        const startCamera = async () => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                if (videoRef.current) {
                    videoRef.current.srcObject = stream;
                }
            } catch (error) {
                console.error('Error accessing the camera:', error);
            }
        };

        if (isCameraOn) {
            startCamera();
        } else {
            stopCamera();
        }

        return () => {
            stopCamera();
        };
    }, [isCameraOn]);

    const stopCamera = () => {
        const videoElement = videoRef.current;
        if (videoElement && videoElement.srcObject) {
            const tracks = videoElement.srcObject.getTracks();
            tracks.forEach(track => track.stop());
            videoElement.srcObject = null;
        }
    };

    const handleCameraOn = () => {
        setIsCameraOn(true);
    };

    const handleCameraOff = () => {
        setIsCameraOn(false);
    };

    const takePhoto = () => {
        const canvas = canvasRef.current;
        const video = videoRef.current;
        if (canvas && video) {
            const context = canvas.getContext('2d');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            context.drawImage(video, 0, 0, canvas.width, canvas.height);
            const dataUrl = canvas.toDataURL('image/png');
            setPhotoUrl(dataUrl);
        }
    };

    return (
        <div>
            <p>Hello</p>
            <video ref={videoRef} autoPlay playsInline style={{ width: '100%', height: 'auto' }} />
            <div>
                <button onClick={handleCameraOn} disabled={isCameraOn}>
                    Camera On
                </button>
                <button onClick={handleCameraOff} disabled={!isCameraOn}>
                    Camera Off
                </button>
                <button onClick={takePhoto} disabled={!isCameraOn}>
                    Take Photo
                </button>
            </div>
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            {photoUrl && (
                <div>
                    <h3>Captured Photo:</h3>
                    <img src={photoUrl} alt="Captured" style={{ width: '100%', height: 'auto' }} />
                    <a href={photoUrl} download="photo.png">
                        Download Photo
                    </a>
                </div>
            )}
        </div>
    );
};

export default Notworking;
