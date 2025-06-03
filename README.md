# Black-Scholes Option Pricing Visualizer

This is a simple web application built with Flask and Plotly to visualize Black-Scholes option pricing using a Geometric Brownian Motion Monte Carlo simulation.

<img width="936" alt="image" src="https://github.com/user-attachments/assets/16a16db0-eb70-4438-a8ab-d815017e0f8f" />

## Features

*   EUROPEAN-STYLE OPTIONS ONLY!!
*   Calculate Theoretical Black-Scholes option prices.
*   Run Monte Carlo simulations (up to 100,000 paths) to estimate option prices.
*   Visualize simulated stock price paths and corresponding option price evolution using interactive Plotly graphs.
*   Display key statistics at expiration, including:
    *   Probability of being In-The-Money (ITM)
    *   90% Confidence Interval for Final Stock Price
    *   Average Final Option Price
    *   Standard Deviation of Final Option Prices

## Setup and Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/jasonli04/BSM-Monte-Carlo-Options-Visualizer
    cd BSM-Monte-Carlo-Options-Visualizer
    ```

2.  **Create a virtual environment (recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

1.  **Make sure your virtual environment is activated (if you created one):**

    ```bash
    source venv/bin/activate
    ```

2.  **Run the Flask application:**

    ```bash
    python app.py
    ```

3.  Open your web browser and go to `http://127.0.0.1:5000/`.

## How it Works

The application uses the Black-Scholes model for calculating the theoretical option price and a Monte Carlo simulation to model potential stock price paths and estimate the option price based on these paths. Plotly is used to create interactive visualizations of the simulated paths.
