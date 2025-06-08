# Black-Scholes Monte Carlo Options Visualizer

You can run this app [here](https://black-scholes-options-visualizer.streamlit.app/)!

A modern web application built with Streamlit and Plotly to visualize Black-Scholes option pricing using a Geometric Brownian Motion Monte Carlo simulation. This tool provides an interactive interface for exploring option pricing and risk metrics through both theoretical calculations and Monte Carlo simulations.

![Image](https://github.com/user-attachments/assets/6d3fd341-d153-46f8-8a94-f7ba4c5a2d65)
![image](https://github.com/user-attachments/assets/b29a5824-2ed3-4f0c-b88a-e364c09e88d6)

## Features

* **European-Style Options Only**
* **Interactive Parameter Controls**
  * Volatility adjustment
  * Underlying price setting
  * Strike price configuration
  * Time to expiration (in years, months, weeks, or days)
  * Risk-free rate adjustment

* **Comprehensive Visualization**
  * Real-time interactive plots using Plotly
  * Call and Put option price evolution
  * Stock price path simulation
  * Confidence intervals for all metrics
  * Theoretical vs. simulated price comparison
  * **Distributions of Final Option Prices**: Histograms showing the frequency (in percentages) of final call and put option prices at expiration, excluding zero-value outcomes for clarity, with a fixed number of buckets for consistent comparison.

* **Detailed Statistics**
  * Theoretical Black-Scholes prices
  * Monte Carlo simulated prices (both discounted and un-discounted)
  * Probability of being In-The-Money (ITM)
  * 90% Confidence Intervals for final prices
  * Price standard deviations
  * Average final prices

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jasonli04/BSM-Monte-Carlo-Options-Visualizer-Streamlit
   cd BSM-Monte-Carlo-Options-Visualizer-Streamlit
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

1. **Make sure your virtual environment is activated:**
   ```bash
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

3. The application will automatically open in your default web browser. If it doesn't, you can access it at the URL shown in the terminal (typically `http://localhost:8501`).

## How it Works

The application combines two powerful approaches to option pricing:

1. **Black-Scholes Model**
   * Provides theoretical option prices based on the famous Black-Scholes formula
   * Assumes log-normal distribution of stock prices
   * Accounts for volatility, time to expiration, risk-free rate, and strike price

2. **Monte Carlo Simulation**
   * Simulates 100,000 potential stock price paths using Geometric Brownian Motion
   * Calculates option prices along each path
   * Provides statistical analysis of the results
   * Helps visualize the range of possible outcomes

The visualization includes:
* Call and Put option price evolution over time
* Stock price paths with confidence intervals
* Theoretical vs. simulated price comparisons
* Interactive hover information for detailed analysis

## Technical Details

* Built with Streamlit for the web interface
* Uses Plotly for interactive visualizations
* Implements vectorized calculations using NumPy
* Monte Carlo simulation with 100,000 paths for accurate results
* Real-time updates as parameters are adjusted

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
