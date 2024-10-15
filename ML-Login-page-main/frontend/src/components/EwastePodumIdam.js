import React from 'react';
import { Route, Routes, useNavigate } from 'react-router-dom';
import Working from './Working';
import UploadForm from './UploadForm'; // Ensure this is correct
import Notworking from './Notworking'
const EwastePodumIdam = () => {
    const navigate = useNavigate();

    return (
        <main className="content">
            <div className="button-container">
                <button onClick={() => navigate('/home/Ewastepodumidam/working')}>Working</button>
                <button onClick={() => navigate('/home/Ewastepodumidam/Notworking')}>Not Working</button>
            </div>
            <Routes>
                <Route path="working" element={<Working />} />
                <Route path="upload" element={<UploadForm />} />
                <Route path="Notworking" element={<Notworking />} /> {/* Ensure this path is correct */}
            </Routes>
        </main>
    );
};

export default EwastePodumIdam;
