// MovableGraph.js

import React, { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

// E-waste generation data for the past six years
const data = [
  { Year: '2022', value: 62.0 }, // Estimated in million tonnes
  { Year: '2021', value: 58.9 },
  { Year: '2020', value: 57.4 },
  { Year: '2019', value: 53.6 },
  { Year: '2018', value: 49.8 },
  { Year: '2017', value: 44.7 },
];

const MovableGraph = () => {
  const [offsetX, setOffsetX] = useState(0);
  const [offsetY, setOffsetY] = useState(0);
  const [isDragging, setIsDragging] = useState(false);

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setOffsetX(e.clientX);
    setOffsetY(e.clientY);
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      const dx = e.clientX - offsetX;
      const dy = e.clientY - offsetY;
      setOffsetX(e.clientX);
      setOffsetY(e.clientY);
      document.getElementById('movable-chart').style.transform = `translate(${dx}px, ${dy}px)`;
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  return (
    <div
      id="movable-chart"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      style={{ cursor: 'grab', width: '100%', height: '400px' }}
    >
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="Year" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Bar dataKey="value" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export default MovableGraph;
