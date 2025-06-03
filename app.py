import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.utils
import json
from flask import Flask, render_template, request, send_file, jsonify
import io
import base64
from BlackScholes import BlackScholes
import numpy as np
import traceback
import sys
from scipy.stats import norm
import plotly.io as pio

pio.renderers.default = 'browser'

app = Flask(__name__)
app.debug = True

# Add vectorized Black-Scholes function
def black_scholes_vectorized(S, K, T, r, sigma, option_type='call'):
    # Ensure inputs are numpy arrays for broadcasting
    S = np.asarray(S)
    K = np.asarray(K)
    T = np.asarray(T)
    r = np.asarray(r)
    sigma = np.asarray(sigma)

    # Avoid division by zero for time=0
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10) # Avoid division by zero if vol is zero

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def make_plot(vol=0.5, underlying_price=100, strike_price=110, time_to_exp=1, risk_free_rate=0.05, option_type='call', time_value=1, time_unit='years'):
    bs = BlackScholes(vol, underlying_price, strike_price, time_to_exp, risk_free_rate)
    theoretical_price = bs.calcPrice(option_type)
    
    # Run Monte Carlo simulation with 1000 paths
    num_simulations = 100000
    mc_price, paths = bs.monte_carlo_simulation(num_simulations, option_type)

    # Create time points for x-axis (initial, in years)
    time_points = np.linspace(0, time_to_exp, paths.shape[1])

    # Calculate total duration in the selected unit for x-axis range
    total_duration_in_unit = time_value # Default to years if not specified elsewhere
    if time_unit == 'months':
        total_duration_in_unit = time_value
        x_axis_title = 'Time (months)'
        x_axis_unit_only = 'months'
    elif time_unit == 'weeks':
        total_duration_in_unit = time_value
        x_axis_title = 'Time (weeks)'
        x_axis_unit_only = 'weeks'
    elif time_unit == 'days':
        total_duration_in_unit = time_value
        x_axis_title = 'Time (days)'
        x_axis_unit_only = 'days'
    else: # years
        total_duration_in_unit = time_value
        x_axis_title = 'Time (years)'
        x_axis_unit_only = 'years'

    # Adjust time_points to be in the selected unit for plotting
    # Currently time_points is 0 to time_to_exp (in years) with paths.shape[1] steps
    # Need to scale this to 0 to total_duration_in_unit
    scaled_time_points = np.linspace(0, total_duration_in_unit, paths.shape[1])

    # Select 20 representative paths (every 5th percentile)
    percentile_indices = np.linspace(0, num_simulations-1, 20, dtype=int)
    representative_paths = paths[percentile_indices]

    # Calculate option prices at each time step for all 1000 paths using the vectorized function
    # Create a grid of parameters for vectorized calculation
    S_grid = paths # Underlying price changes over time and per path
    K_grid = np.full_like(paths, strike_price) # Strike price is constant
    # Time remaining: T - time_points (broadcasts time_points across paths)
    T_grid = time_to_exp - time_points # Needs broadcasting
    T_grid = np.maximum(T_grid, 1e-10) # Avoid T=0 issues
    r_grid = np.full_like(paths, risk_free_rate) # Risk-free rate is constant
    sigma_grid = np.full_like(paths, vol) # Volatility is constant

    # Calculate option prices using the vectorized function
    # Need to handle broadcasting carefully or loop over time steps for T_grid
    # Let's loop over time steps, but use the vectorized BS for each step
    option_prices = np.zeros_like(paths)
    for t_idx in range(paths.shape[1]):
         time_rem = time_to_exp - time_points[t_idx]
         if time_rem > 1e-10: # Use a small epsilon instead of 0 for time remaining
              option_prices[:, t_idx] = black_scholes_vectorized(
                  S=paths[:, t_idx],
                  K=strike_price,
                  T=time_rem,
                  r=risk_free_rate,
                  sigma=vol,
                  option_type=option_type
              )
         else:
             # At expiration, calculate payoff (vectorized)
             if option_type.lower() == 'call':
                 option_prices[:, t_idx] = np.maximum(paths[:, t_idx] - strike_price, 0)
             else:
                 option_prices[:, t_idx] = np.maximum(strike_price - paths[:, t_idx], 0)

    # Get representative option price paths corresponding to the representative stock paths
    representative_option_prices = option_prices[percentile_indices]

    # Calculate 90% confidence intervals
    stock_lower_bound_path = np.percentile(paths, 5, axis=0)
    stock_upper_bound_path = np.percentile(paths, 95, axis=0)
    option_lower_bound = np.percentile(option_prices, 5, axis=0)
    option_upper_bound = np.percentile(option_prices, 95, axis=0)

    # Calculate statistics at expiration
    final_stock_prices = paths[:, -1]
    final_option_prices = option_prices[:, -1]

    # Probability of being In-The-Money (ITM) at expiration
    if option_type.lower() == 'call':
        # For a call option, ITM if final stock price > strike price
        itm_count = np.sum(final_stock_prices > strike_price)
    else: # put option
        # For a put option, ITM if final stock price < strike price
        itm_count = np.sum(final_stock_prices < strike_price)
    prob_itm = (itm_count / num_simulations) * 100.0

    # 90% Confidence Interval for Final Stock Price
    final_stock_ci_lower = np.percentile(final_stock_prices, 5)
    final_stock_ci_upper = np.percentile(final_stock_prices, 95)

    # Average and Standard Deviation of final option prices
    avg_final_option_price = np.mean(final_option_prices)
    std_final_option_price = np.std(final_option_prices)

    # DEBUG prints (optional, but good for verification)
    # print("DEBUG: time_points shape:", time_points.shape)
    # print("DEBUG: representative_paths shape:", representative_paths.shape)
    # print("DEBUG: representative_option_prices shape:", representative_option_prices.shape)
    # print("DEBUG: representative_option_prices sample (first 3 paths, first 10 steps):")
    # print(representative_option_prices[:3, :10])
    # print("DEBUG: average stock path shape:", np.mean(paths, axis=0).shape)
    # print("DEBUG: average option price path shape:", np.mean(option_prices, axis=0).shape)
    # print("DEBUG: strike_price:", strike_price)
    # print("DEBUG: theoretical_price:", theoretical_price)
    # print("DEBUG: stock_lower_bound_path sample (first 10 steps):", stock_lower_bound_path[:10])
    # print("DEBUG: stock_upper_bound_path sample (first 10 steps):", stock_upper_bound_path[:10])
    # print("DEBUG: option_lower_bound shape:", option_lower_bound.shape)
    # print("DEBUG: option_upper_bound sample (first 10 steps):", option_upper_bound[:10])


    # Create subplot figure - 1 row, 2 columns
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Stock Price Paths (20 Representative Paths)',
                                      'Option Price Evolution (20 Representative Paths)'),
                        horizontal_spacing=0.1)

    # Define a list of distinct colors for the 20 paths
    colors = plotly.colors.qualitative.Plotly # Using a built-in Plotly color sequence

    # Add stock price paths (20 representative) to the first subplot
    for i, path in enumerate(representative_paths):
        trace_name = f'Stock Path {percentile_indices[i]}'
        fig.add_trace(
            go.Scatter(
                x=scaled_time_points, y=path, mode='lines', line=dict(color=colors[i % len(colors)], width=1.5), showlegend=False, # Use a color from the list
                xaxis='x', yaxis='y',
                name=trace_name,
                hovertemplate=f'<b>{trace_name}</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
            ),
            row=1, col=1
        )

    # Add average stock path to the first subplot
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=np.mean(paths, axis=0), mode='lines', name='Average Stock Path', line=dict(color='red', width=2),
            xaxis='x', yaxis='y',
            hovertemplate=f'<b>Average Stock Path</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add stock price confidence interval bounds to the first subplot
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=stock_lower_bound_path, mode='lines', name='90% CI (Stock - Lower)', line=dict(color='#444', dash=None, width=1.5), showlegend=False,
            xaxis='x', yaxis='y',
            hovertemplate=f'<b>90% CI (Stock - Lower)</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=stock_upper_bound_path, mode='lines', name='90% CI (Stock - Upper)', line=dict(color='#444', dash=None, width=1.5), showlegend=False,
            xaxis='x', yaxis='y',
            hovertemplate=f'<b>90% CI (Stock - Upper)</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add strike price line to the first subplot
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=np.full_like(scaled_time_points, strike_price), mode='lines', name='Strike Price', line=dict(color='green', dash='dash', width=2),
            xaxis='x', yaxis='y',
            hovertemplate=f'<b>Strike Price</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add option price paths (20 representative) to the second subplot
    for i, path in enumerate(representative_option_prices):
         trace_name = f'Option Path {percentile_indices[i]}'
         fig.add_trace(
             go.Scatter(
                 x=scaled_time_points, y=path, mode='lines', line=dict(color=colors[i % len(colors)], width=1.5), showlegend=False, # Use the same color as the corresponding stock path
                 xaxis='x2', yaxis='y2',
                 name=trace_name,
                 hovertemplate=f'<b>{trace_name}</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
             ),
             row=1, col=2
         )

    # Add average option price path to the second subplot
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=np.mean(option_prices, axis=0), mode='lines', name='Average Option Price', line=dict(color='red', width=2),
            xaxis='x2', yaxis='y2',
            hovertemplate=f'<b>Average Option Price</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=2
    )

    # Add option price confidence interval bounds to the second subplot
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=option_lower_bound, mode='lines', name='90% CI (Option - Lower)', line=dict(color='#444', dash=None, width=1.5), showlegend=False,
            xaxis='x2', yaxis='y2',
            hovertemplate=f'<b>90% CI (Option - Lower)</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=option_upper_bound, mode='lines', name='90% CI (Option - Upper)', line=dict(color='#444', dash=None, width=1.5), showlegend=False,
            xaxis='x2', yaxis='y2',
            hovertemplate=f'<b>90% CI (Option - Upper)</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=2
    )

    # Add theoretical price line to the second subplot
    # Note: Theoretical price is constant over time for a fixed time_to_exp in BS model, but plotting as a line for visual consistency
    fig.add_trace(
         go.Scatter(
             x=scaled_time_points, y=np.full_like(scaled_time_points, theoretical_price), mode='lines', name='Theoretical Price', line=dict(color='green', dash='dash', width=2),
             xaxis='x2', yaxis='y2',
             hovertemplate=f'<b>Theoretical Price</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
         ),
         row=1, col=2
     )

    # Update layout for both subplots
    fig.update_layout(
        title_text='Black-Scholes Monte Carlo Simulation', # Use title_text for overall title
        height=600, # Increased height slightly
        showlegend=False, # Remove legend
        hovermode='closest', # Show hover information for the closest data point
        margin=dict(l=80, r=40, t=80, b=80) # Adjust margins to provide space for labels
    )

    fig.update_xaxes(title_text=x_axis_title, row=1, col=1,
                     range=[0, total_duration_in_unit], fixedrange=True) # Prevent panning outside the defined range
    # Auto-scale Y for stock price, include 0 and strike
    stock_y_min = np.min(representative_paths)
    stock_y_max = np.max(representative_paths)
    # Adjust y-axis range to include confidence intervals
    stock_ci_min = np.min(stock_lower_bound_path)
    stock_ci_max = np.max(stock_upper_bound_path)
    y1_range_adjusted = [min(0, stock_y_min * 0.8, strike_price * 0.8, stock_ci_min * 0.8), max(stock_y_max * 1.2, strike_price * 1.2, stock_ci_max * 1.2)]
    fig.update_yaxes(title_text='Stock Price ($)', row=1, col=1, range=y1_range_adjusted)

    fig.update_xaxes(title_text=x_axis_title, row=1, col=2,
                     range=[0, total_duration_in_unit], fixedrange=True) # Prevent panning outside the defined range
    # Auto-scale Y for option price, include 0 and theoretical price
    option_y_min = np.min(representative_option_prices)
    option_y_max = np.max(representative_option_prices)
    # Adjust y-axis range to include confidence intervals
    option_ci_min = np.min(option_lower_bound)
    option_ci_max = np.max(option_upper_bound)
    y2_range_adjusted = [min(0, option_y_min * 0.8, theoretical_price * 0.8, option_ci_min * 0.8), max(option_y_max * 1.2, theoretical_price * 1.2, option_ci_max * 1.2)]
    fig.update_yaxes(title_text='Option Price ($)', row=1, col=2, range=y2_range_adjusted)

    # Generate HTML div and script for the figure
    # include_plotlyjs='cdn' ensures the JS library is loaded from a CDN
    plot_div = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

    # Return the HTML div string, prices, and statistics
    return plot_div, theoretical_price, mc_price, prob_itm, final_stock_ci_lower, final_stock_ci_upper, avg_final_option_price, std_final_option_price # Return plot_div (HTML string)

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            # Get parameters from form
            vol = float(request.form['volatility'])
            underlying_price = float(request.form['underlying_price'])
            strike_price = float(request.form['strike_price'])
            time_value = float(request.form['time_value'])
            time_unit = request.form['time_unit']
            risk_free_rate = float(request.form['risk_free_rate'])
            option_type = request.form['option_type']

            # Convert time to expiration to years
            if time_unit == 'months':
                time_to_exp = time_value / 12.0
            elif time_unit == 'weeks':
                time_to_exp = time_value / 52.0
            elif time_unit == 'days':
                time_to_exp = time_value / 365.0
            else: # assume years
                time_to_exp = time_value

            # Input validation
            if vol <= 0:
                raise ValueError("Volatility must be positive")
            if underlying_price <= 0:
                raise ValueError("Underlying price must be positive")
            if strike_price <= 0:
                raise ValueError("Strike price must be positive")
            if time_to_exp <= 0:
                raise ValueError("Time to expiration must be positive")

            # Call make_plot, it now returns plot_div and statistics
            plot_div, theoretical_price, mc_price, prob_itm, final_stock_ci_lower, final_stock_ci_upper, avg_final_option_price, std_final_option_price = make_plot(
                vol=vol,
                underlying_price=underlying_price,
                strike_price=strike_price,
                time_to_exp=time_to_exp,
                risk_free_rate=risk_free_rate,
                option_type=option_type,
                time_value=time_value,
                time_unit=time_unit
            )

            # Pass plot_div, prices, and statistics to the template
            return render_template('index.html', plot_div=plot_div, theoretical_price=theoretical_price, mc_price=mc_price,
                                   vol=vol, underlying_price=underlying_price, strike_price=strike_price,
                                   risk_free_rate=risk_free_rate, option_type=option_type, time_value=time_value, time_unit=time_unit,
                                   prob_itm=prob_itm, final_stock_ci_lower=final_stock_ci_lower, final_stock_ci_upper=final_stock_ci_upper,
                                   avg_final_option_price=avg_final_option_price, std_final_option_price=std_final_option_price)
        else:
            # Generate initial visualization with default values
            # Call make_plot, it now returns plot_div and statistics
            plot_div, theoretical_price, mc_price, prob_itm, final_stock_ci_lower, final_stock_ci_upper, avg_final_option_price, std_final_option_price = make_plot(time_value=1, time_unit='years') # Pass default time values

            # Pass plot_div, prices, and statistics to the template
            return render_template('index.html',
                                plot_div=plot_div,
                                theoretical_price=theoretical_price,
                                mc_price=mc_price,
                                vol=0.5,
                                underlying_price=100,
                                strike_price=110,
                                risk_free_rate=0.05,
                                option_type='call',
                                time_value=1,
                                time_unit='years',
                                prob_itm=prob_itm,
                                final_stock_ci_lower=final_stock_ci_lower,
                                final_stock_ci_upper=final_stock_ci_upper,
                                avg_final_option_price=avg_final_option_price,
                                std_final_option_price=std_final_option_price)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}\n\nFull traceback:\n{traceback.format_exc()}"
        print(error_message, file=sys.stderr)
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True) 