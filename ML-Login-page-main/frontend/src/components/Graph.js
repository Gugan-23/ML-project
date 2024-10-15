import React from 'react';

const Graph = () => {
    return (
        <div>
            <h2>Graph</h2>
            <img 
                src="/cor.PNG" 
                alt="Correlation Graph" 
                style={{ width: '100%', height: 'auto', marginBottom: '40px' }} // Increased margin
            />
            <img 
                src="/Count.PNG" 
                alt="Count Graph" 
                style={{ width: '100%', height: 'auto', marginBottom: '40px' }} // Increased margin
            />
            <img 
                src="/confusion.PNG" 
                alt="Confusion Graph" 
                style={{ width: '100%', height: 'auto' }} 
            />
        </div>
    );
};

export default Graph;
