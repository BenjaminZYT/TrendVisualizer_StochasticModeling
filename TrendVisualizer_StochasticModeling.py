import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.signal import savgol_filter
from statsmodels.api import OLS, add_constant
import plotly.express as px
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash import callback_context

def get_djia():
    url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
    try:
        tables = pd.read_html(url)
        for table in tables:
            if 'Symbol' in table.columns:
                return table, table['Symbol'].tolist()
    except Exception as e:
        print(f"Error fetching DJIA table: {e}")
        return pd.DataFrame(), []

    return pd.DataFrame(), []

# Generate ticker selection for Dash dropdown
djia_df, djia_tickers = get_djia()
dropdown_options = [
    {'label': '^IRX', 'value': '^IRX'},
    {'label': '^TYX', 'value': '^TYX'},
    {'label': '^SPX', 'value': '^SPX'}
    ] + [{'label': ticker, 'value': ticker} for ticker in djia_tickers]

app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.Div([     
        html.H1("Trend Visualizer with Stochastic Modeling"),
        "What to Know and How to Use? ",
        html.A("Click here.", href="https://drive.google.com/file/d/1DKxyd6YG0073q9YqPgGkyptlofVtA0pN/view?usp=sharing", target="_blank")
    ], style={'margin-bottom': '20px', 'font-weight': 'bold'}),

    dcc.Dropdown(
        id='ticker-dropdown',
        options=dropdown_options,
        value=None,
        clearable=True,
        placeholder="Select a ticker"
    ),

    dcc.Input(
        id='ticker-input',
        type='text',
        value='',
        placeholder='Or enter a ticker',
        style={'display': 'inline-block', 'margin-left': '10px'}
    ),

    html.Button('Reset', id='reset-button', n_clicks=0),
    html.Div(id='error-message', style={'color': 'red'}),

    dcc.RadioItems(
        id='ohlc',
        options=[
            {'label': 'Open', 'value': 'Open'},
            {'label': 'High', 'value': 'High'},
            {'label': 'Low', 'value': 'Low'},
            {'label': 'Close', 'value': 'Close'},
            {'label': 'Adj Close', 'value': 'Adj Close'},
            {'label': 'Volume', 'value': 'Volume'}
        ],
        value='Close',
        labelStyle={'display': 'inline-block'}
    ),
    html.Label('How far back (in years)?', style={'margin-right': '10px'}),
    dcc.Input(
        id='backdate',
        type='number',
        min=0.25,
        max=4,
        step=0.25,
        value=2,
        style={'margin-left': '10px'}
    ),

    html.Button('Go', id='go-button', n_clicks=0, style={'margin-left': '10px'}),
    html.Button('Download CSV', id='download-button', n_clicks=0, style={'margin-left': '10px'}),
    
    html.Div([
        html.H3("Trend Plot"),
        dcc.Graph(id='graph', style={'height': '80vh'})
    ]),
    
    html.Div([
        html.H3("Model Plot"),
        dcc.Dropdown(
            id='model-dropdown',
            options=[
                {'label': 'Geometric Brownian Motion Model (good for modeling stock prices)', 'value': 'gbm1'},
                {'label': 'Merton Jump-Diffusion Model (for modeling stock prices short term, with instances of shock)', 'value': 'mjd1'},
                {'label': 'Mean Reverting Model (normally used to model commodity prices)', 'value': 'mr1'},
                {'label': 'Cox-Ingersoll-Ross Model (normally used for modeling interest rates and bond prices)', 'value': 'cir1'}
            ],
            value=None,
            clearable=True,
            placeholder="Select a model"
        ),
        html.Button('View Model', id='view-model-button', n_clicks=0, style={'margin-left': '10px'}),
        dcc.Graph(id='model-graph', style={'height': '80vh'})
    ]),

    dcc.Download(id='download-dataframe')
])
    
def validate_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(period='1d')
        if hist_data.empty or {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}.difference(hist_data.columns):
            return True
        return False
    except Exception as e:
        print(f"Error validating ticker: {e}")
        return False


def SGdrift_component(selected_data, order):
    max_window_length = min(101, len(selected_data))
    if max_window_length % 2 == 0:
        max_window_length -= 1
    
    window_length = max_window_length if max_window_length > 1 else 3
    trend = savgol_filter(selected_data, window_length=window_length, polyorder=order)
    
    return trend


# Computing a parameter that will be useful for several models below, starting with OU
def estimate_kappa(ticker_data):
    log_returns, _, _, _ = log_data(ticker_data)

    lagged_returns = log_returns.shift(1).dropna()
    current_returns = log_returns[1:].dropna()

    regression_data = pd.DataFrame({
        'Current': current_returns,
        'Lagged': lagged_returns
    }).dropna()

    X = add_constant(regression_data['Lagged'])
    y = regression_data['Current']
    model = OLS(y, X).fit()

    kappa = model.params['Lagged']
    
    return kappa

def log_data(ticker_data):
    log_returns = np.log(ticker_data / ticker_data.shift(1)).dropna()
    threshold = log_returns.mean() + 2 * log_returns.std()
    jumps = log_returns[np.abs(log_returns) > threshold]
    jump_intensity = len(jumps) / len(log_returns)
    mean_jump_size = jumps.mean()
    std_jump_size = jumps.std()

    return log_returns, jump_intensity, mean_jump_size, std_jump_size



def GBM(ticker_data):
    choice_prices = ticker_data
    dates = choice_prices.index
    log_returns = np.log(choice_prices / choice_prices.shift(1)).dropna()
    Mu = log_returns.mean()
    Sigma = log_returns.std()
    T = len(choice_prices)
    dt = 1
    gbm_prices = [choice_prices[0]]
    for _ in range(1, T):
        drift = (Mu - 0.5 * Sigma**2) * dt
        shock = Sigma * np.sqrt(dt) * np.random.normal()
        price = gbm_prices[-1] * np.exp(drift + shock)
        gbm_prices.append(price)
    gbm_prices = pd.Series(gbm_prices, index=dates)
    return gbm_prices

def MR(ticker_data):
    kappa = estimate_kappa(ticker_data)
    mu = ticker_data.mean()
#     sigma = ticker_data.std()
    sigma = ticker_data.pct_change().std()
    S0 = ticker_data.iloc[0]
    dt = 1 #1/(10*kappa)
    N = len(ticker_data)
    S = np.zeros(N)
    S[0] = S0
    W = np.random.normal(0, np.sqrt(dt), N)
    for i in range(1, N):
        S[i] = S[i-1] + kappa * (mu - S[i-1]) * dt + (sigma/(1.1*sigma)) * W[i] #Originally just sigma*W[i]
    return pd.Series(S, index=ticker_data.index)

def CIR(ticker_data):
    kappa = 1/(estimate_kappa(ticker_data))
    mu = ticker_data.mean()
#     sigma = ticker_data.std()
#     sigma = ticker_data.pct_change().std()
    sigma = ticker_data.pct_change().tail(7).std()
    r0 = ticker_data.iloc[0]
    dt = 1 #1/kappa
    T = len(ticker_data)
    cir_rates = [r0]
    for _ in range(1, T):
#         drift = kappa * (mu - cir_rates[-1]) * dt
        drift = (1/mu**2.5) * (mu - cir_rates[-1]) * dt
#         shock = sigma * np.sqrt(cir_rates[-1]) * np.sqrt(dt) * np.random.normal()
        shock = 10*sigma * np.sqrt(cir_rates[-1]) * np.sqrt(dt) * np.random.normal()
        rate = cir_rates[-1] + drift + shock
        cir_rates.append(rate)
    cir_rates = pd.Series(cir_rates, index=ticker_data.index)
    return cir_rates

def MJD(ticker_data):
    mu = ticker_data.mean()
#     sigma = ticker_data.std()
    sigma = ticker_data.pct_change().std()
    log_returns, jump_intensity, mean_jump_size, std_jump_size = log_data(ticker_data)
    S0 = ticker_data.iloc[0]
    T = len(ticker_data)
    dt = 0.00001
    mjd_prices = [S0]
    for _ in range(1, T):
        drift = (mu - 0.5 * sigma**2) * dt
#         shock = sigma * np.sqrt(dt) * np.random.normal()
        shock = (100*sigma) * np.sqrt(dt) * np.random.normal()
        jump = (np.random.poisson(jump_intensity) * 
                (np.exp(mean_jump_size + std_jump_size * np.random.normal()) - 1))
        price = mjd_prices[-1] * np.exp(1*(drift + shock)) * (1 + jump)
        mjd_prices.append(price)
    mjd_prices = pd.Series(mjd_prices, index=ticker_data.index)
    return mjd_prices

# Global variable to store data
global_data = None
global_ticker = None
global_ohlc = None

@app.callback(
    [Output('graph', 'figure'),
     Output('model-graph', 'figure'),
     Output('error-message', 'children'),
     Output('ticker-dropdown', 'value'),
     Output('ticker-input', 'value')],
    [Input('go-button', 'n_clicks'),
     Input('reset-button', 'n_clicks'),
     Input('view-model-button', 'n_clicks')],
    [State('ticker-dropdown', 'value'),
     State('ticker-input', 'value'),
     State('backdate', 'value'),
     State('ohlc', 'value'),
     State('model-dropdown', 'value')]
)
def update_plots(go_clicks, reset_clicks, view_model_clicks, dropdown_value, input_value, n_years, ohlc, selected_model):
    global global_data, global_ticker, global_ohlc
    ctx = callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    trend_fig = dash.no_update
    model_fig = dash.no_update

    if triggered_id == 'reset-button':
        global_data = None
        global_ticker = None
        global_ohlc = None
        return {}, {}, '', None, ''

    if triggered_id == 'go-button':
        if dropdown_value and input_value:
            if dropdown_value != input_value.upper():
                return {}, {}, 'Tickers do not match. Please ensure both tickers are the same.', None, None

        ticker = dropdown_value if dropdown_value else input_value.upper()
        if not ticker:
            return {}, {}, 'Please provide a ticker.', None, None

        if not validate_ticker(ticker):
            return {}, {}, 'Invalid ticker or no data available.', None, None

        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * n_years)
        
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            if data.empty:
                return {}, {}, f'No data found for {ticker}.', None, None
            global_data = data
            global_ticker = ticker
            global_ohlc = ohlc
        except Exception as e:
            return {}, {}, f'Error retrieving data for {ticker}: {str(e)}', None, None

        selected_data = data[ohlc]
        trend_line_1 = SGdrift_component(selected_data, 4)
        trend_line_2 = SGdrift_component(selected_data, 5)

        plot_data = pd.DataFrame({
            'Date': selected_data.index,
            'Price or Volume': selected_data.values,
            'Trend Line': trend_line_1,
            'Alternative Trend Line': trend_line_2
        })

        trend_fig = px.line(plot_data, x='Date', y=['Price or Volume', 'Alternative Trend Line', 'Trend Line'],
                            labels={'value': '', 'variable': ''},
                            title=f'Trend of "{ohlc}" for {ticker}')
        
        trend_fig.add_annotation(
            text="By Benjamin Z.Y. Teoh @ July 2024 @ Alpharetta, GA",
            xref="paper", yref="paper",
            x=0.5, y=1.1,
            showarrow=False,
            font=dict(size=9)
        )

        trend_fig.update_traces(mode='lines', line=dict(color='lightblue'), selector=dict(name='Price or Volume'))
        trend_fig.update_traces(mode='lines', line=dict(color='red'), selector=dict(name='Trend Line'))
        trend_fig.update_traces(mode='lines', line=dict(color='blue'), selector=dict(name='Alternative Trend Line'))

        trend_fig.update_layout(xaxis_title='', yaxis_title='')
        trend_fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5))
        trend_fig.update_layout(legend=dict(traceorder='reversed'))

    if triggered_id == 'view-model-button':
        if global_data is None:
            return trend_fig, {}, 'Please generate the trend plot first.', None, None

        selected_data = global_data[ohlc]
        ticker_symbol = global_ticker
        ohlcv = global_ohlc

        if selected_model == 'gbm1':
            model_prices = GBM(selected_data)
            title = 'Geometric Brownian Motion Model'
        elif selected_model == 'mjd1':
            model_prices = MJD(selected_data)
            title = 'Merton Jump-Diffusion Model'
        elif selected_model == 'mr1':
            model_prices = MR(selected_data)
            title = 'Mean Reverting Model'
        elif selected_model == 'cir1':
            model_prices = CIR(selected_data)
            title = 'Cox-Ingersoll-Ross Model'
        else:
            return trend_fig, {}, 'Please select a valid model.', None, None

        model_data = pd.DataFrame({
            'Date': selected_data.index,
            'Actual Price or Volume': selected_data.values,
            'Modeled Price or Volume': model_prices
        })

        model_fig = px.line(model_data, x='Date', y=['Actual Price or Volume', 'Modeled Price or Volume'],
                            labels={'value': '', 'variable': ''},
                            title=f'{title} on {ticker_symbol} "{ohlc}"')
        
        long_term_mean = selected_data.mean()
        ltm = round(long_term_mean, 2)
        
        model_fig.add_trace(go.Scatter(x=model_data['Date'], y=[long_term_mean]*len(model_data['Date']),
                               mode='lines', line=dict(color='gray', dash='dash'),
                               name=f'Long-term Mean {ltm}'))

        model_fig.add_annotation(
            text="By Benjamin Z.Y. Teoh @ July 2024 @ Alpharetta, GA",
            xref="paper", yref="paper",
            x=0.5, y=1.1,
            showarrow=False,
            font=dict(size=9)
        )

        model_fig.update_traces(mode='lines', line=dict(color='blue'), selector=dict(name='Actual'))
        model_fig.update_traces(mode='lines', line=dict(color='lightblue'), selector=dict(name='Modeled'))

        model_fig.update_layout(xaxis_title='', yaxis_title='')
        model_fig.update_layout(legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5))
        model_fig.update_layout(legend=dict(traceorder='reversed'))

    return trend_fig, model_fig, '', None, ''

@app.callback(
    Output('download-dataframe', 'data'),
    Input('download-button', 'n_clicks')
)
def download_csv(n_clicks):
    if n_clicks > 0 and global_data is not None:
        filename = f"{global_ticker}_data.csv"
        return dcc.send_data_frame(global_data.to_csv, filename)
    return None

if __name__ == "__main__":
    app.run_server(debug=True)




