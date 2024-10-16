import React, { useState } from 'react';

const Purchase = () => {
    // Sample products data with real-world products
    const products = [
        {
            id: 1,
            name: 'Apple iPhone 14',
            price: 799.99,
            imageUrl: 'data:image/jpeg;base64,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',
        },
        {
            id: 2,
            name: 'Samsung Galaxy S23',
            price: 999.99,
            imageUrl: 'https://imgs.search.brave.com/a0jDWuQ2c31QuKNYBk-OAeA1bmLGwyqP6C00ZKgpxEk/rs:fit:500:0:0:0/g:ce/aHR0cHM6Ly9mZG4y/LmdzbWFyZW5hLmNv/bS92di9iaWdwaWMv/c2Ftc3VuZy1nYWxh/eHktYTU1LmpwZw',
        },
        {
            id: 3,
            name: 'boAt WH-1000XM4 Headphones',
            price: 349.99,
            imageUrl: 'https://encrypted-tbn3.gstatic.com/shopping?q=tbn:ANd9GcQEHW3r9zmypvPZLn99MfCcn5RtH1l68P4YFXSBINUQufdCEz60BtEfJsAkBdl41BX7KiwFJSrpKPCU75NXOX-_CKWg1Wh2WqA71fquo173Q59RP1WFYznXHw&usqp=CAc',
        },
    ];

    const [cart, setCart] = useState([]); // State for the shopping cart

    const handleBuyNow = (product) => {
        // Handle the Buy Now action
        console.log(`Purchasing ${product.name} for $${product.price}`);
        // You can add your purchase logic here (e.g., redirect to a checkout page)
    };

    const handleAddToCart = (product) => {
        // Handle the Add to Cart action
        setCart((prevCart) => [...prevCart, product]); // Add product to cart
        console.log(`${product.name} added to cart`);
    };

    return (
        <div className="purchase-container">
            <h2>Available Products</h2>
            <div className="product-list">
                {products.map((product) => (
                    <div className="product-card" key={product.id}>
                        <img src={product.imageUrl} alt={product.name} className="product-image" />
                        <h3>{product.name}</h3>
                        <p>Price: ${product.price.toFixed(2)}</p>
                        <button onClick={() => handleBuyNow(product)}>Buy Now</button>
                        <button onClick={() => handleAddToCart(product)}>Add to Cart</button>
                    </div>
                ))}
            </div>
            <div className="cart-info">
                <h3>Shopping Cart</h3>
                <ul>
                    {cart.map((item, index) => (
                        <li key={index}>{item.name} - ${item.price.toFixed(2)}</li>
                    ))}
                </ul>
                <p>Total items in cart: {cart.length}</p>
            </div>
        </div>
    );
};

export default Purchase;
