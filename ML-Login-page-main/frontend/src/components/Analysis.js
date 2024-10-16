import React from 'react';
import { Route, Routes, useNavigate } from 'react-router-dom';
import UploadForm from './UploadForm'; // Ensure this is correct
import Notworking from './Notworking'
import Graph from './Graph'
import Model from './Model'
import Predictip from './Predictip' 
const Analysis = () => {
    const navigate = useNavigate();

    return (
        <main className="content">
            <div className="button-container">
                <button onClick={() => navigate('/home/ewastegraph/Model')}>E waste</button>
                <button onClick={() => navigate('/home/ewastegraph/graph')}>Graph</button>
                <button onClick={() => navigate('/home/ewastegraph/Help')}>Help</button>

            </div>
            <Routes>
            <Route path="Model" element={<Model />} />
                <Route path="graph" element={<Graph />} />
                <Route path="upload" element={<UploadForm />} />
                <Route path="Help" element={<Predictip />} />

                <Route path="Notworking" element={<Notworking />} /> {/* Ensure this path is correct */}
            </Routes>
        </main>
    );
};

export default Analysis;
