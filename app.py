import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import numpy as np
from scipy.stats import norm
from BlackScholes import BlackScholes

def create_base_figure(call_prob_itm=0, put_prob_itm=0):
    """Create the base figure with subplots."""
    return make_subplots(
        rows=3, cols=2,
        subplot_titles=('Stock Price Paths', 'Call Option Price Evolution', 'Put Option Price Evolution',
                        f'Final Call Price Distribution (Prob ITM: {call_prob_itm:.1f}%)', 
                        f'Final Put Price Distribution (Prob ITM: {put_prob_itm:.1f}%)'),
        specs=[[{"colspan": 2}, None], [{}, {}], [{}, {}]],
        vertical_spacing=0.1,
        shared_xaxes=False # We need independent x-axes for histograms
    )

def add_option_paths(fig, time_points, paths, prices, option_type, row, col, colors):
    """Add option price paths to the figure."""
    for i, path in enumerate(paths):
        trace_name = f'{option_type} Path {i}'
        fig.add_trace(
            go.Scatter(
                x=time_points, y=path, mode='lines',
                line=dict(color=colors[i % len(colors)], width=1.5),
                showlegend=False,
                name=trace_name,
                hovertemplate=f'<b>{trace_name}</b><br>Time: %{{x:.2f}} years<br>Price: $%{{y:.2f}}<extra></extra>'
            ),
            row=row, col=col
        )

def add_average_path(fig, time_points, prices, name, row, col):
    """Add average price path to the figure."""
    fig.add_trace(
        go.Scatter(
            x=time_points, y=np.mean(prices, axis=0),
            mode='lines', name=name,
            line=dict(color='red', width=2),
            hovertemplate=f'<b>{name}</b><br>Time: %{{x:.2f}} years<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=row, col=col
    )

def add_confidence_intervals(fig, time_points, lower_bound, upper_bound, name, row, col):
    """Add confidence interval bounds to the figure."""
    fig.add_trace(
        go.Scatter(
            x=time_points, y=lower_bound,
            mode='lines', name=f'90% CI ({name})',
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2),
            showlegend=False,
            hovertemplate=f'<b>90% CI ({name} - Lower)</b><br>Time: %{{x:.2f}} years<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=row, col=col
    )
    fig.add_trace(
        go.Scatter(
            x=time_points, y=upper_bound,
            mode='lines', name=f'90% CI ({name})',
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2),
            showlegend=False,
            hovertemplate=f'<b>90% CI ({name} - Upper)</b><br>Time: %{{x:.2f}} years<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=row, col=col
    )

def add_theoretical_price(fig, time_points, price, name, row, col):
    """Add theoretical price line to the figure."""
    fig.add_trace(
        go.Scatter(
            x=time_points, y=np.full_like(time_points, price),
            mode='lines', name=f'Theoretical {name} Price',
            line=dict(color='green', dash='dash', width=2),
            hovertemplate=f'<b>Theoretical {name} Price</b><br>Time: %{{x:.2f}} years<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=row, col=col
    )

def update_figure_layout(fig, x_axis_title):
    """Update the figure layout with consistent styling."""
    fig.update_layout(
        title_text='Black-Scholes Monte Carlo Simulation',
        height=1500, # Increased height for more vertical space
        showlegend=False,
        hovermode='closest',
        margin=dict(l=80, r=40, t=80, b=80),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )
    
    # Update x-axes for all subplots
    fig.update_xaxes(title_text=x_axis_title, row=1, col=1)  # Stock price paths
    fig.update_xaxes(title_text=x_axis_title, row=2, col=1)  # Call option price evolution
    fig.update_xaxes(title_text=x_axis_title, row=2, col=2)  # Put option price evolution
    
    # Update grid styling for all axes
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

def calculate_option_prices(paths, strike_price, time_points, time_to_exp, risk_free_rate, vol):
    """Calculate option prices along the paths."""
    call_prices = np.zeros_like(paths)
    put_prices = np.zeros_like(paths)
    
    for t_idx in range(paths.shape[1]):
        time_rem = time_to_exp - time_points[t_idx]
        if time_rem > 1e-10:
            bs = BlackScholes(vol, paths[:, t_idx], strike_price, time_rem, risk_free_rate)
            call_prices[:, t_idx] = bs.calcPrice('call')
            put_prices[:, t_idx] = bs.calcPrice('put')
        else:
            # At expiration, calculate payoffs
            call_prices[:, t_idx] = np.maximum(paths[:, t_idx] - strike_price, 0)
            put_prices[:, t_idx] = np.maximum(strike_price - paths[:, t_idx], 0)
    
    return call_prices, put_prices

def make_plot(vol=0.5, underlying_price=100, strike_price=110, time_to_exp=1, risk_free_rate=0.05, time_value=1, time_unit='years'):
    """Create the main plot with all components."""
    bs = BlackScholes(vol, underlying_price, strike_price, time_to_exp, risk_free_rate)
    
    # Calculate theoretical prices
    call_theoretical_price = bs.calcPrice('call')
    put_theoretical_price = bs.calcPrice('put')
    
    # Run Monte Carlo simulation
    num_simulations = 100000
    call_mc_price, put_mc_price, paths = bs.monte_carlo_simulation(num_simulations)

    # Create time points
    time_points = np.linspace(0, time_to_exp, paths.shape[1])
    scaled_time_points = np.linspace(0, time_value, paths.shape[1])

    # Select representative paths
    percentile_indices = np.linspace(0, num_simulations-1, 20, dtype=int)
    representative_paths = paths[percentile_indices]

    # Calculate option prices
    call_prices, put_prices = calculate_option_prices(paths, strike_price, time_points, time_to_exp, risk_free_rate, vol)
    representative_call_prices = call_prices[percentile_indices]
    representative_put_prices = put_prices[percentile_indices]

    # Calculate confidence intervals
    stock_lower_bound = np.percentile(paths, 5, axis=0)
    stock_upper_bound = np.percentile(paths, 95, axis=0)
    call_lower_bound = np.percentile(call_prices, 5, axis=0)
    call_upper_bound = np.percentile(call_prices, 95, axis=0)
    put_lower_bound = np.percentile(put_prices, 5, axis=0)
    put_upper_bound = np.percentile(put_prices, 95, axis=0)

    # Calculate probabilities
    final_stock_prices = paths[:, -1]
    final_call_prices = call_prices[:, -1]
    final_put_prices = put_prices[:, -1]

    # Filter out zero prices for histogram visualization
    final_call_prices_filtered = final_call_prices[final_call_prices > 0]
    final_put_prices_filtered = final_put_prices[final_put_prices > 0]

    # Dynamically set x-axis range for call histogram
    if len(final_call_prices_filtered) > 0: 
        call_x_min = np.percentile(final_call_prices_filtered, 1)
        call_x_max = np.percentile(final_call_prices_filtered, 99)
    else:
        call_x_min = 0
        call_x_max = 1 

    # Dynamically set x-axis range for put histogram
    if len(final_put_prices_filtered) > 0: 
        put_x_min = np.percentile(final_put_prices_filtered, 1)
        put_x_max = np.percentile(final_put_prices_filtered, 99)
    else:
        put_x_min = 0
        put_x_max = 1 

    # Define number of histogram bins
    num_histogram_bins = 15 # Set to 15 buckets

    # Calculate specific bin sizes for call and put
    call_bin_size = (call_x_max - call_x_min) / num_histogram_bins if (call_x_max - call_x_min) > 0 else 1
    put_bin_size = (put_x_max - put_x_min) / num_histogram_bins if (put_x_max - put_x_min) > 0 else 1

    # Define xbins for call and put separately
    call_xbins = dict(start=call_x_min, end=call_x_max, size=call_bin_size)
    put_xbins = dict(start=put_x_min, end=put_x_max, size=put_bin_size)

    call_itm_count = np.sum(final_stock_prices > strike_price)
    put_itm_count = np.sum(final_stock_prices < strike_price)
    call_prob_itm = (call_itm_count / num_simulations) * 100.0
    put_prob_itm = (put_itm_count / num_simulations) * 100.0

    final_stock_ci_lower = np.percentile(final_stock_prices, 5)
    final_stock_ci_upper = np.percentile(final_stock_prices, 95)
    final_call_ci_lower = np.percentile(final_call_prices, 5)
    final_call_ci_upper = np.percentile(final_call_prices, 95)
    final_put_ci_lower = np.percentile(final_put_prices, 5)
    final_put_ci_upper = np.percentile(final_put_prices, 95)

    discounted_call_price = np.mean(np.maximum(final_stock_prices - strike_price, 0)) * np.exp(-risk_free_rate * time_to_exp)
    discounted_put_price = np.mean(np.maximum(strike_price - final_stock_prices, 0)) * np.exp(-risk_free_rate * time_to_exp)
    undiscounted_call_price = np.mean(np.maximum(final_stock_prices - strike_price, 0))
    undiscounted_put_price = np.mean(np.maximum(strike_price - final_stock_prices, 0))

    # Create figure
    fig = create_base_figure(call_prob_itm=call_prob_itm, put_prob_itm=put_prob_itm)
    colors = plotly.colors.qualitative.Plotly

    # Add paths and bounds for each subplot
    add_option_paths(fig, scaled_time_points, representative_paths, paths, 'Stock', 1, 1, colors)
    add_option_paths(fig, scaled_time_points, representative_call_prices, call_prices, 'Call', 2, 1, colors)
    add_option_paths(fig, scaled_time_points, representative_put_prices, put_prices, 'Put', 2, 2, colors)

    add_average_path(fig, scaled_time_points, paths, 'Average Stock Price', 1, 1)
    add_average_path(fig, scaled_time_points, call_prices, 'Average Call Price', 2, 1)
    add_average_path(fig, scaled_time_points, put_prices, 'Average Put Price', 2, 2)

    add_confidence_intervals(fig, scaled_time_points, stock_lower_bound, stock_upper_bound, 'Stock', 1, 1)
    add_confidence_intervals(fig, scaled_time_points, call_lower_bound, call_upper_bound, 'Call', 2, 1)
    add_confidence_intervals(fig, scaled_time_points, put_lower_bound, put_upper_bound, 'Put', 2, 2)

    add_theoretical_price(fig, scaled_time_points, strike_price, 'Strike', 1, 1)
    add_theoretical_price(fig, scaled_time_points, call_theoretical_price, 'Call', 2, 1)
    add_theoretical_price(fig, scaled_time_points, put_theoretical_price, 'Put', 2, 2)

    # Update layout
    x_axis_title = f'Time ({time_unit})'
    update_figure_layout(fig, x_axis_title)

    # Update x-axes for price evolution plots (now in row 2)
    fig.update_xaxes(range=[0, time_value], fixedrange=False, minallowed=0, maxallowed=time_value, row=2, col=1)
    fig.update_xaxes(range=[0, time_value], fixedrange=False, minallowed=0, maxallowed=time_value, row=2, col=2)

    # Update y-axes for price evolution plots (now in row 2)
    call_y_min = np.min(representative_call_prices)
    call_y_max = np.max(representative_call_prices)
    call_ci_min = np.min(call_lower_bound)
    call_ci_max = np.max(call_upper_bound)
    call_y_range = [min(0, call_y_min * 0.8, call_theoretical_price * 0.8, call_ci_min * 0.8),
                   max(call_y_max * 1.2, call_theoretical_price * 1.2, call_ci_max * 1.2)]
    fig.update_yaxes(title_text='Call Option Price ($)', row=2, col=1, range=call_y_range)

    put_y_min = np.min(representative_put_prices)
    put_y_max = np.max(representative_put_prices)
    put_ci_min = np.min(put_lower_bound)
    put_ci_max = np.max(put_upper_bound)
    put_y_range = [min(0, put_y_min * 0.8, put_theoretical_price * 0.8, put_ci_min * 0.8),
                  max(put_y_max * 1.2, put_theoretical_price * 1.2, put_ci_max * 1.2)]
    fig.update_yaxes(title_text='Put Option Price ($)', row=2, col=2, range=put_y_range)

    # Update y-axis for stock price (now in row 1)
    stock_y_min = np.min(representative_paths)
    stock_y_max = np.max(representative_paths)
    stock_ci_min = np.min(stock_lower_bound)
    stock_ci_max = np.max(stock_upper_bound)
    stock_y_range = [min(0, stock_y_min * 0.8, strike_price * 0.8, stock_ci_min * 0.8),
                    max(stock_y_max * 1.2, strike_price * 1.2, stock_ci_max * 1.2)]
    fig.update_yaxes(title_text='Stock Price ($)', row=1, col=1, range=stock_y_range)

    # Apply range constraints to the stock path x-axis (row 1, col 1)
    fig.update_xaxes(range=[0, time_value], fixedrange=False, minallowed=0, maxallowed=time_value, row=1, col=1)

    # Add histograms for final option prices (remain in row 3)
    fig.add_trace(
        go.Histogram(
            x=final_call_prices, # Use unfiltered data
            name='Final Call Price Distribution', # Updated name
            marker_color='blue', 
            opacity=0.75,
            hovertemplate='Price: %{x}<br>Frequency: %{y:.2f}%<extra></extra>',
            xbins=call_xbins, # Use specific xbins for call
            histnorm='percent' # Normalize to show percentages
        ),
        row=3, col=1
    )

    fig.update_xaxes(title_text='Final Call Price ($)', row=3, col=1, range=[call_x_min, call_x_max])
    fig.update_yaxes(title_text='Frequency (%)', row=3, col=1)

    fig.add_trace(
        go.Histogram(
            x=final_put_prices, # Use unfiltered data
            name='Final Put Price Distribution', # Updated name
            marker_color='orange', 
            opacity=0.75,
            hovertemplate='Price: %{x}<br>Frequency: %{y:.2f}%<extra></extra>',
            xbins=put_xbins, # Use specific xbins for put
            histnorm='percent' # Normalize to show percentages
        ),
        row=3, col=2
    )

    fig.update_xaxes(title_text='Final Put Price ($)', row=3, col=2, range=[put_x_min, put_x_max])
    fig.update_yaxes(title_text='Frequency (%)', row=3, col=2)

    return fig, call_theoretical_price, call_mc_price, put_theoretical_price, put_mc_price, \
           call_prob_itm, put_prob_itm, final_stock_ci_lower, final_stock_ci_upper, \
           np.mean(final_call_prices), np.std(final_call_prices), \
           np.mean(final_put_prices), np.std(final_put_prices), \
           final_call_ci_lower, final_call_ci_upper, final_put_ci_lower, final_put_ci_upper, \
           discounted_call_price, discounted_put_price, \
           undiscounted_call_price, undiscounted_put_price

def main():
    st.set_page_config(page_title="Black-Scholes Monte Carlo Options Visualizer", layout="wide")

    # CSS to make the spinner icon white and larger
    st.markdown("""
    <style>
    /* Target the SVG icon within the spinner */
    .stSpinner > div > div > svg {
        fill: white !important; /* Make the spinner icon white */
        width: 75px !important; /* Make it larger */
        height: 75px !important; /* Make it larger */
    }
    /* Ensure the text is white */
    .stSpinner > div > div > div {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("Black-Scholes Monte Carlo Options Visualizer")

    # Create sidebar for inputs
    st.sidebar.header("Input Parameters")

    with st.sidebar.form("input_form"):
        # Input parameters
        vol = st.slider("Volatility", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        underlying_price = st.number_input("Underlying Price ($)", min_value=1.0, value=100.0, step=1.0)
        strike_price = st.number_input("Strike Price ($)", min_value=1.0, value=110.0, step=1.0)
        time_value = st.number_input("Time Value", min_value=0.01, value=1.0, step=0.01)
        time_unit = st.selectbox("Time Unit", ["years", "months", "weeks", "days"])
        risk_free_rate = st.slider("Risk-Free Rate (%)", min_value=0.0, max_value=20.0, value=5.0, step=0.1) / 100.0

        submitted = st.form_submit_button("Run Simulation")

    # Convert time to expiration to years
    if time_unit == 'months':
        time_to_exp = time_value / 12.0
    elif time_unit == 'weeks':
        time_to_exp = time_value / 52.0
    elif time_unit == 'days':
        time_to_exp = time_value / 365.0
    else: # assume years
        time_to_exp = time_value

    if submitted:
        try:
            # Show loading spinner while running simulations
            with st.spinner("Running Monte Carlo Simulations... This may take a few seconds while we simulate 100,000 paths"):
                # Generate plot and statistics
                fig, call_theoretical_price, call_mc_price, put_theoretical_price, put_mc_price, \
                    call_prob_itm, put_prob_itm, final_stock_ci_lower, final_stock_ci_upper, \
                    avg_final_call_price, std_final_call_price, \
                    avg_final_put_price, std_final_put_price, \
                    final_call_ci_lower, final_call_ci_upper, final_put_ci_lower, final_put_ci_upper, \
                    discounted_call_price, discounted_put_price, \
                    undiscounted_call_price, undiscounted_put_price = make_plot(
                    vol=vol,
                    underlying_price=underlying_price,
                    strike_price=strike_price,
                    time_to_exp=time_to_exp,
                    risk_free_rate=risk_free_rate,
                    time_value=time_value,
                    time_unit=time_unit
                )

            # Display the plot
            st.plotly_chart(fig, use_container_width=True)

            # Create two columns for statistics
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Call Option Statistics")
                st.metric("Theoretical Price", f"${call_theoretical_price:.2f}")
                st.metric("Monte Carlo Simulated Price (Discounted)", f"${call_mc_price:.2f}")
                st.metric("Monte Carlo Simulated Price (Un-discounted)", f"${undiscounted_call_price:.2f}")
                st.metric("Probability ITM", f"{call_prob_itm:.1f}%")
                st.metric("Final Price (90% CI)", f"${final_call_ci_lower:.2f} - ${final_call_ci_upper:.2f}")
                st.metric("Price Standard Deviation", f"${std_final_call_price:.2f}")

            with col2:
                st.subheader("Put Option Statistics")
                st.metric("Theoretical Price", f"${put_theoretical_price:.2f}")
                st.metric("Monte Carlo Simulated Price (Discounted)", f"${put_mc_price:.2f}")
                st.metric("Monte Carlo Simulated Price (Un-discounted)", f"${undiscounted_put_price:.2f}")
                st.metric("Probability ITM", f"{put_prob_itm:.1f}%")
                st.metric("Final Price (90% CI)", f"${final_put_ci_lower:.2f} - ${final_put_ci_upper:.2f}")
                st.metric("Price Standard Deviation", f"${std_final_put_price:.2f}")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        # Initial display with default values or no display until submitted
        st.info("Adjust parameters in the sidebar and click 'Run Simulation' to view results.")

if __name__ == '__main__':
    main() 