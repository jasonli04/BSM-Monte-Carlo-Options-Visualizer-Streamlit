import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import numpy as np
from scipy.stats import norm
from BlackScholes import BlackScholes

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

def make_plot(vol=0.5, underlying_price=100, strike_price=110, time_to_exp=1, risk_free_rate=0.05, time_value=1, time_unit='years'):
    bs = BlackScholes(vol, underlying_price, strike_price, time_to_exp, risk_free_rate)
    
    # Calculate both call and put theoretical prices
    call_theoretical_price = bs.calcPrice('call')
    put_theoretical_price = bs.calcPrice('put')
    
    # Run Monte Carlo simulation for both call and put
    num_simulations = 100000
    call_mc_price, put_mc_price, paths = bs.monte_carlo_simulation(num_simulations)

    # Create time points for x-axis (initial, in years)
    time_points = np.linspace(0, time_to_exp, paths.shape[1])

    # Calculate total duration in the selected unit for x-axis range
    total_duration_in_unit = time_value
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

    scaled_time_points = np.linspace(0, total_duration_in_unit, paths.shape[1])

    # Select 20 representative paths
    percentile_indices = np.linspace(0, num_simulations-1, 20, dtype=int)
    representative_paths = paths[percentile_indices]

    # Calculate option prices for both call and put
    call_option_prices = np.zeros_like(paths)
    put_option_prices = np.zeros_like(paths)
    
    for t_idx in range(paths.shape[1]):
        time_rem = time_to_exp - time_points[t_idx]
        if time_rem > 1e-10:
            call_option_prices[:, t_idx] = black_scholes_vectorized(
                S=paths[:, t_idx],
                K=strike_price,
                T=time_rem,
                r=risk_free_rate,
                sigma=vol,
                option_type='call'
            )
            put_option_prices[:, t_idx] = black_scholes_vectorized(
                S=paths[:, t_idx],
                K=strike_price,
                T=time_rem,
                r=risk_free_rate,
                sigma=vol,
                option_type='put'
            )
        else:
            # At expiration, calculate payoffs
            call_option_prices[:, t_idx] = np.maximum(paths[:, t_idx] - strike_price, 0)
            put_option_prices[:, t_idx] = np.maximum(strike_price - paths[:, t_idx], 0)

    # Get representative option price paths
    representative_call_prices = call_option_prices[percentile_indices]
    representative_put_prices = put_option_prices[percentile_indices]

    # Calculate confidence intervals
    stock_lower_bound_path = np.percentile(paths, 5, axis=0)
    stock_upper_bound_path = np.percentile(paths, 95, axis=0)
    call_lower_bound = np.percentile(call_option_prices, 5, axis=0)
    call_upper_bound = np.percentile(call_option_prices, 95, axis=0)
    put_lower_bound = np.percentile(put_option_prices, 5, axis=0)
    put_upper_bound = np.percentile(put_option_prices, 95, axis=0)

    # Calculate statistics at expiration
    final_stock_prices = paths[:, -1]
    final_call_prices = call_option_prices[:, -1]
    final_put_prices = put_option_prices[:, -1]

    # Calculate final payoffs
    final_call_payoffs = np.maximum(final_stock_prices - strike_price, 0)
    final_put_payoffs = np.maximum(strike_price - final_stock_prices, 0)

    # Calculate discounted present values using final payoffs
    discounted_call_price = np.mean(final_call_payoffs) * np.exp(-risk_free_rate * time_to_exp)
    discounted_put_price = np.mean(final_put_payoffs) * np.exp(-risk_free_rate * time_to_exp)
    
    # Calculate un-discounted final prices
    undiscounted_call_price = np.mean(final_call_payoffs)
    undiscounted_put_price = np.mean(final_put_payoffs)

    # Probability of being ITM
    call_itm_count = np.sum(final_stock_prices > strike_price)
    put_itm_count = np.sum(final_stock_prices < strike_price)
    call_prob_itm = (call_itm_count / num_simulations) * 100.0
    put_prob_itm = (put_itm_count / num_simulations) * 100.0

    # 90% Confidence Interval for Final Stock Price
    final_stock_ci_lower = np.percentile(final_stock_prices, 5)
    final_stock_ci_upper = np.percentile(final_stock_prices, 95)

    # 90% Confidence Interval for Final Call Option Price
    final_call_ci_lower = np.percentile(final_call_prices, 5)
    final_call_ci_upper = np.percentile(final_call_prices, 95)

    # 90% Confidence Interval for Final Put Option Price
    final_put_ci_lower = np.percentile(final_put_prices, 5)
    final_put_ci_upper = np.percentile(final_put_prices, 95)

    # Create subplot figure - 2 rows, 2 columns (bottom row spans both columns)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Call Option Price Evolution', 'Put Option Price Evolution', 'Stock Price Paths'),
        specs=[[{}, {}], [{"colspan": 2}, None]],
        vertical_spacing=0.1,
        shared_xaxes=True # Share x-axes vertically
    )

    # Define colors for paths
    colors = plotly.colors.qualitative.Plotly

    # Add call option paths
    for i, path in enumerate(representative_call_prices):
        trace_name = f'Call Path {percentile_indices[i]}'
        fig.add_trace(
            go.Scatter(
                x=scaled_time_points, y=path, mode='lines',
                line=dict(color=colors[i % len(colors)], width=1.5),
                showlegend=False,
                name=trace_name,
                hovertemplate=f'<b>{trace_name}</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
            ),
            row=1, col=1
        )

    # Add put option paths
    for i, path in enumerate(representative_put_prices):
        trace_name = f'Put Path {percentile_indices[i]}'
        fig.add_trace(
            go.Scatter(
                x=scaled_time_points, y=path, mode='lines',
                line=dict(color=colors[i % len(colors)], width=1.5),
                showlegend=False,
                name=trace_name,
                hovertemplate=f'<b>{trace_name}</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
            ),
            row=1, col=2
        )

    # Add average paths for call options
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=np.mean(call_option_prices, axis=0),
            mode='lines', name='Average Call Price',
            line=dict(color='red', width=2),
            hovertemplate=f'<b>Average Call Price</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add call option confidence interval bounds
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=call_lower_bound,
            mode='lines', name='90% CI (Call)',
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2),
            showlegend=False,
            hovertemplate=f'<b>90% CI (Call - Lower)</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=call_upper_bound,
            mode='lines', name='90% CI (Call)',
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2),
            showlegend=False,
            hovertemplate=f'<b>90% CI (Call - Upper)</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add average paths for put options
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=np.mean(put_option_prices, axis=0),
            mode='lines', name='Average Put Price',
            line=dict(color='red', width=2),
            hovertemplate=f'<b>Average Put Price</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=2
    )

    # Add put option confidence interval bounds
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=put_lower_bound,
            mode='lines', name='90% CI (Put)',
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2),
            showlegend=False,
            hovertemplate=f'<b>90% CI (Put - Lower)</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=2
    )

    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=put_upper_bound,
            mode='lines', name='90% CI (Put)',
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2),
            showlegend=False,
            hovertemplate=f'<b>90% CI (Put - Upper)</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=2
    )

    # Add theoretical prices
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=np.full_like(scaled_time_points, call_theoretical_price),
            mode='lines', name='Theoretical Call Price',
            line=dict(color='green', dash='dash', width=2),
            hovertemplate=f'<b>Theoretical Call Price</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=np.full_like(scaled_time_points, put_theoretical_price),
            mode='lines', name='Theoretical Put Price',
            line=dict(color='green', dash='dash', width=2),
            hovertemplate=f'<b>Theoretical Put Price</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=1, col=2
    )

    # Add stock price paths to the bottom row
    for i, path in enumerate(representative_paths):
        trace_name = f'Stock Path {percentile_indices[i]}'
        fig.add_trace(
            go.Scatter(
                x=scaled_time_points, y=path, mode='lines',
                line=dict(color=colors[i % len(colors)], width=1.5),
                showlegend=False,
                name=trace_name,
                hovertemplate=f'<b>{trace_name}</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
            ),
            row=2, col=1
        )

    # Add average stock path
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=np.mean(paths, axis=0),
            mode='lines', name='Average Stock Path',
            line=dict(color='red', width=2),
            hovertemplate=f'<b>Average Stock Path</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=2, col=1
    )

    # Add stock price confidence interval bounds
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=stock_lower_bound_path,
            mode='lines', name='90% CI (Stock)',
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2),
            showlegend=False,
            hovertemplate=f'<b>90% CI (Stock - Lower)</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=stock_upper_bound_path,
            mode='lines', name='90% CI (Stock)',
            line=dict(color='rgba(255, 255, 255, 0.7)', width=2),
            showlegend=False,
            hovertemplate=f'<b>90% CI (Stock - Upper)</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=2, col=1
    )

    # Add strike price line
    fig.add_trace(
        go.Scatter(
            x=scaled_time_points, y=np.full_like(scaled_time_points, strike_price),
            mode='lines', name='Strike Price',
            line=dict(color='green', dash='dash', width=2),
            hovertemplate=f'<b>Strike Price</b><br>Time: %{{x:.2f}} {x_axis_unit_only}<br>Price: $%{{y:.2f}}<extra></extra>'
        ),
        row=2, col=1
    )

    # Update layout with dark background
    fig.update_layout(
        title_text='Black-Scholes Monte Carlo Simulation',
        height=800,  # Increased height for better visibility
        showlegend=False,
        hovermode='closest',
        margin=dict(l=80, r=40, t=80, b=80),
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white')
    )

    # Update x-axes (only the bottom one needs a title)
    fig.update_xaxes(title_text=x_axis_title, row=2, col=1)

    # Apply range constraints to the shared x-axis
    fig.update_xaxes(range=[0, total_duration_in_unit], fixedrange=False, minallowed=0, maxallowed=total_duration_in_unit)

    # Update axes with white grid lines
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.2)')

    # Update y-axes
    # For call options
    call_y_min = np.min(representative_call_prices)
    call_y_max = np.max(representative_call_prices)
    call_ci_min = np.min(call_lower_bound)
    call_ci_max = np.max(call_upper_bound)
    call_y_range = [min(0, call_y_min * 0.8, call_theoretical_price * 0.8, call_ci_min * 0.8),
                   max(call_y_max * 1.2, call_theoretical_price * 1.2, call_ci_max * 1.2)]
    fig.update_yaxes(title_text='Call Option Price ($)', row=1, col=1, range=call_y_range)

    # For put options
    put_y_min = np.min(representative_put_prices)
    put_y_max = np.max(representative_put_prices)
    put_ci_min = np.min(put_lower_bound)
    put_ci_max = np.max(put_upper_bound)
    put_y_range = [min(0, put_y_min * 0.8, put_theoretical_price * 0.8, put_ci_min * 0.8),
                  max(put_y_max * 1.2, put_theoretical_price * 1.2, put_ci_max * 1.2)]
    fig.update_yaxes(title_text='Put Option Price ($)', row=1, col=2, range=put_y_range)

    # For stock price
    stock_y_min = np.min(representative_paths)
    stock_y_max = np.max(representative_paths)
    stock_ci_min = np.min(stock_lower_bound_path)
    stock_ci_max = np.max(stock_upper_bound_path)
    stock_y_range = [min(0, stock_y_min * 0.8, strike_price * 0.8, stock_ci_min * 0.8),
                    max(stock_y_max * 1.2, strike_price * 1.2, stock_ci_max * 1.2)]
    fig.update_yaxes(title_text='Stock Price ($)', row=2, col=1, range=stock_y_range)

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
        vol = st.slider("Volatility", min_value=0.01, max_value=2.0, value=0.5, step=0.01)
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