import React from 'react';
import { Link, Route, Routes } from 'react-router-dom';
import AboutUs from './AboutUs';
import ContactUs from './ContactUs';
import Profile from './Profile';
import EwasteGraph from './EwasteGraph';
import EwastePodumIdam from './EwastePodumIdam';
import Purchase from './Purchase'
import './Home.css';
import Working from './Working'; 
import Analysis from './Analysis';
import Notworking from './Notworking'; 
import Model from './Model'
import Graph from './Graph'


const Home = () => {
    return (
        <div className="home-container">
            <aside className="sidebar">
                <h3>Dashboard</h3>
                <ul>
                    <li><Link to="aboutus">About Us</Link></li>
                    <li><Link to="contactus">Contact Us</Link></li>
                    <li><Link to="profile">Profile</Link></li>
                    <li><Link to="ewastegraph">E-waste Analysis Graph</Link></li>
                    <li><Link to="EwastePodumIdam">E-waste Podum Idam</Link></li>
                    <li><Link to="Purchase">Purchase</Link></li>
                    
                    
               </ul>
            </aside>
            <main className="content">
                <Routes>
                    <Route path="aboutus" element={<AboutUs />} />
                    <Route path="contactus" element={<ContactUs />} />
                    <Route path="profile" element={<Profile />} />
                    <Route path="ewastegraph/*" element={<EwasteGraph />} />
                    <Route path="ewastepodumidam/*" element={<EwastePodumIdam />} />
                    <Route path="working" element={<Working />} />
                    <Route path="Notworking" element={<Notworking />} />
                    <Route path="analysis" element={<Analysis />} />
                    <Route path="Graph" element={<Graph />} />
                    <Route path="Model" element={<Model />} />
                    <Route path="Purchase" element={<Purchase />} />
                </Routes>
            </main>
        </div>
    );
};

export default Home;
