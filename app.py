import logging

import dash
from dash.dependencies import Input, Output, State

import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

import dash_table

import numpy as np

import plotly.express as px
import plotly.graph_objs as go

import shap

import vaex

from src import load_data_sources, sensor_data

# #################################
# Configure the logging
# #################################

import dash_daq as daq


logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')

# #################################
# Load all of the data
# #################################

df_test_expl, df_test_trans, df_test_final, explainer, df_airplane = load_data_sources()
# These are the features for the Keras model (scaled, shifted features)
features = df_test_trans.get_column_names(regex='^feat')

# For the app we use the sensor names like T2/Total temperature at fan inlet
sensor_names_short = [sensor[0] for sensor in sensor_data if sensor[0] in df_test_trans]
sensor_names_long = [sensor[1] for sensor in sensor_data if sensor[0] in df_test_trans]
sensor_names_unit = [sensor[2] for sensor in sensor_data if sensor[0] in df_test_trans]
sensor_names_full = [name if unit == '--' else f'{name} [{unit}]' for name, unit in zip(sensor_names_long, sensor_names_unit)]

engine_ids_left = df_airplane.left.to_numpy()
engine_ids_right = df_airplane.right.to_numpy()

risk_classes = ['low', 'medium', 'high']

# #################################
# Functions that create the figures
# #################################

def create_feature_importance_figure(expl):
    df = vaex.from_arrays(expl=expl,
                          sensor_names_short=sensor_names_short,
                          sensor_names_full=sensor_names_full).sort('expl')

    fig = px.bar(df.to_pandas_df(),
                 x='expl',
                 y='sensor_names_short',
                 orientation='h',
                 custom_data=['sensor_names_full'],
                 labels={'expl': 'Relative importance', 'sensor_names_short': 'Sensor'})
    # fig.layout.update(showlegend=False, title='Feature importance')
    fig.data[0]['hovertemplate'] = '<br>Sensor: %{customdata}<br>Relative importance: %{x:.3f}<extra></extra>'
    return fig


def create_explain_sensor_figure(x, y, w, sensor, sequence_length=50):
    sensor_index = sensor_names_short.index(sensor)
    n_cycles = x.shape[0]

    y_min = y.min()
    y_max = y.max()

    line_spec = go.scatter.Line(color='blue', width=2)
    hovertemplate = 'Cycle: %{x}<br>Sensor value: %{y:.3f}<br>Shap value: %{customdata:.3f}<extra></extra>'
    line = go.Scatter(x=x, y=y, mode='lines', line=line_spec, name=None, hovertemplate=hovertemplate, customdata=w)

    line_spec = go.bar.marker.Line(cmax=3, cmid=0.3, cmin=0.0, color=w, colorscale='Blues')
    bars_spec = go.bar.Marker(cmax=2, cmid=0.3, cmin=0.0, color=w, colorscale='Blues', opacity=0.3, line=line_spec)
    bars = go.Bar(x=x, y=[y_max]*n_cycles, marker=bars_spec, width=1, hoverinfo='skip')

    layout = go.Layout(xaxis=go.layout.XAxis(title='Cycle', range=(n_cycles-sequence_length, n_cycles)),
                       yaxis=go.layout.YAxis(title=sensor_names_short[sensor_index], range=(y_min, y_max)),
                       title=f'{sensor_names_full[sensor_index]}',
                       showlegend=False,
                       plot_bgcolor="#F9F9F9",
                       paper_bgcolor="#F9F9F9"
                       )
    fig = go.Figure(data=[line, bars], layout=layout)
    return fig


# #################################
# Functions that compute the data
# #################################

def get_engine_info(engine_number):
    if engine_number is None:
        md = '_Click on image to select an engine_'
    else:
        df_tmp = df_test_final[df_test_final.unit_number == engine_number]
        pred, gt, current = df_tmp[['RUL_pred', 'RUL_gt', 'current_cycle']].values.astype(int)[0]
        position = "left" if engine_number in engine_ids_left else "right"

        md = f'''
            ### Selected Engine

            - Position - {position}
            - Engine id - {engine_number}
            - Current cycle: {current}
            - Estimated RUL: {pred}
            - True RUL: {gt}
            '''
    return md


def get_shap_values(engine_number):
    df_tmp = df_test_trans[df_test_trans.unit_number == engine_number]
    seq_one = df_tmp[features][-1:].values.data.transpose(1, 0, 2)
    shap_values = explainer.shap_values(seq_one)[0]
    shap_values = np.abs(shap_values)
    return shap_values


def compute_feature_importances(shap_values):
    '''For the relative feature improtance bar chart'''
    return (shap_values[0]).sum(axis=0)


def compute_sensor_data(engine_number, sensor, shap_values, scaled=True):
    index = sensor_names_short.index(sensor)

    df_tmp = df_test_expl[df_test_expl.unit_number == engine_number]
    x = np.arange(df_tmp.shape[0]) + 0.5
    if scaled:
        sensor = f'minmax_scaled_{sensor}'
    y = df_tmp[sensor].values

    # The weight - the Shap values
    w = shap_values[0, :, index]

    # For the full range
    n_cycles = x.shape[0]
    w = np.concatenate((np.zeros(n_cycles - 50), w))

    return x, y, w


# #################################
# Instantiate the Dash app
# #################################

external_stylesheets = []
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Dash+Vaex: Predictive maintenance"
server = app.server   # used by gunicorn in production mode

# #################################
# Initial Conditions and Settings
# #################################

# The table data
RUL_LEVEL_MEDIUM, RUL_LEVEL_HIGH = 40, 20


def risk(rul):
    return (rul <= RUL_LEVEL_MEDIUM).where((rul <= RUL_LEVEL_HIGH).where('high', 'medium'), 'low')


df_airplane['risk'] = risk(df_airplane.RUL_pred_shortest)
df_test_final['risk'] = risk(df_test_final.RUL_pred)
df_airplane = df_airplane.sort('RUL_pred_shortest')[['tail_number', 'RUL_pred_shortest', 'risk', 'left', 'right']]
table_data = df_airplane.to_pandas_df().to_dict(orient='records')

# Initial conditions
tail_number_init = df_airplane[:1]['tail_number'].tolist()[0]
engine_number_init = 3
sensor_init = sensor_names_short[0]
scaled_init = 'yes'
engine_info_init = get_engine_info(engine_number_init)
shap_values_init = get_shap_values(engine_number=engine_number_init)
expl_init = compute_feature_importances(shap_values=shap_values_init)
x_init, y_init, w_init = compute_sensor_data(engine_number=engine_number_init, sensor=sensor_init, shap_values=shap_values_init, scaled=scaled_init)

# About the app - description [TBC]
about_md = '''
### Vaex and Dash: Predictive maintenance

An example of an interactive dashboard which shows when a fault is expected to
occur in a group of jet engines. The data is preprocessed with
[Vaex](https://github.com/vaexio/vaex/), which is also used to create a complete
ML pipeline. The model itself is built with [Keras](https://keras.io/), and
[SHAP](https://github.com/slundberg/shap) is used to offer some insight and
interpretability of the model results.
All of it is put together via [Dash](https://plotly.com/dash/).

Check out this [GitHub repo](https://github.com/vaexio/dash-predictive-maintenance-app)
which contans the source code of this dashboard as well as the model creation notebooks.
'''

# Additional settings
graph_config = {"modeBarButtonsToRemove": ['lasso2d', 'select2d']}

table_data_formatting = [
    {'if': {'filter_query': f'{{RUL_pred_shortest}} <= {RUL_LEVEL_HIGH}'},
     'backgroundColor': '#fff3f3',
     'color': 'black',
    },
    {'if': {'filter_query': f'{{RUL_pred_shortest}} <= {RUL_LEVEL_MEDIUM} && {{RUL_pred_shortest}} > {RUL_LEVEL_HIGH}'},
     'backgroundColor': '#f9f5ea',
     'color': 'black'
    },
]

# #################################
# Application Layout
# #################################

app.layout = html.Div(className='app-body', children=[

    # Store
    dcc.Store(id='tail-number', data=tail_number_init, storage_type='local'),
    dcc.Store(id='engine-number', data=engine_number_init),
    dcc.Store(id='shap-values', data=shap_values_init),

    # App title and logos
    html.Div(className='row', children=[
        html.Div(className='twelve columns', children=[
            html.Div(style={'float': 'left'}, children=[
                html.H1('Vaex & Dash: Predictive maintenance example'),
                html.H4('Predicting Aircraft turbine engine Remaining Useful Life (RUL)')
            ]),
            html.Div(style={'float': 'right'}, children=[
                html.A(html.Img(src=app.get_asset_url('vaex-logo.png'), style={'float': 'right', 'height': '35px', 'margin-top': '20px'}), href='https://vaex.io'),
                html.A(html.Img(src=app.get_asset_url('dash-logo.png'), style={'float': 'right', 'height': '55px'}), href='https://dash.plot.ly')
            ]),
        ]),
    ]),
    html.Hr(),
    # The visuals
    html.Div(className='row', children=[
        html.Div(className='four columns container', children=[
            html.Img(src='https://img.icons8.com/ios/452/jet-engine.png', style={'height': '250px', 'opacity': '0.3'})
        ]),
        html.Div(className='four columns container', children=[
            daq.Gauge(
                id='gauge-high',
                label="High risk engine failure",
                color="#ff5c7b",
                min=0,
                max=len(df_test_final),
                value=(df_test_final.RUL_pred <= RUL_LEVEL_HIGH).sum()
            )
        ]),
        html.Div(className='four columns container', children=[
            daq.Gauge(
                id='gauge-medium',
                label="Medium risk engine failure",
                color="#ffc550",
                min=0,
                max=len(df_test_final),
                value=(df_test_final.RUL_pred <= RUL_LEVEL_MEDIUM).sum(),
            )
        ]),
    ]),
    html.Div(className='row', children=[
        html.Div(className='three columns pretty_container', children=[
            html.H3('Airplane fleet'),
            dash_table.DataTable(id='airplane_table',
                                columns=[{'name': 'tail number', 'id': 'tail_number'}, {'name': 'RUL', 'id': 'RUL_pred_shortest'}, {'name': 'risk', 'id': 'risk'}],
                                data=table_data,
                                style_table={'height': '100%', 'overflowY': 'auto'},
                                style_data_conditional=table_data_formatting)
        ]),
        html.Div(className='nine columns container', children=[
            html.Div(className='row', children=[
                html.Div(className='twelve columns pretty_container', children=[
                    html.Div([
                        html.Button(className="engine engine-left engine-active", id="button-left-engine"),
                        html.Button(className="engine engine-right", id="button-right-engine"),
                        html.Img(src="assets/airplane/warning-red.svg", id="high-left", className="warning-indicator"),
                        html.Img(src="assets/airplane/warning-red.svg", id="high-right", className="warning-indicator"),
                        html.Img(src="assets/airplane/warning-yellow.svg", id="medium-left", className="warning-indicator"),
                        html.Img(src="assets/airplane/warning-yellow.svg", id="medium-right", className="warning-indicator"),
                        html.Img(src="assets/airplane/airplane2.svg", id="airplane-image", style={'width': '100%'}, useMap='#map'),
                    ])
                ])
            ]),
            html.Div(className='row', children=[
                html.Div(className='four columns pretty_container', children=[
                    dcc.Loading(className='loader', id='loading', type='default', children=[
                        dcc.Markdown(id='engine-info', children=engine_info_init)
                    ]),
                ]),

                html.Div(className='eight columns pretty_container', children=[
                    dcc.Markdown(id='feature-importance-label', children='''
                    ### Sensor importances

                    _Click on a bar to select a sensor_
                    '''),
                    dcc.Graph(id='sensor-importance-figure',
                            figure=create_feature_importance_figure(expl=expl_init),
                            config=graph_config)
                ]),
            ]),
            html.Div(className='row', children=[
                html.Div(className='twelve columns pretty_container', children=[
                    dcc.Markdown(id='explainer-label', children='### Sensor explorer'),
                    dcc.Dropdown(id='sensor-list',
                                placeholder='Select a sensor',
                                options=[{'label': value, 'value': value} for value in sensor_names_short],
                                value=sensor_init,
                                multi=False),
                    dcc.Checklist(id='scale', options=[{'label': 'Normalize', 'value': 'yes'}], value=[scaled_init]),
                    dcc.Graph(id='explainer-figure',
                            figure=create_explain_sensor_figure(x=x_init, y=y_init, w=w_init, sensor=sensor_init),
                            config=graph_config)
                ]),
            ]),
        ]),
    ]),
    # Credits
    html.Hr(),
    dcc.Markdown(children=about_md),
])


# #################################
# Callbacks
# #################################

@app.callback(
        [
            Output('tail-number', 'data'),
            Output('engine-number', 'data'),
            Output('button-left-engine', 'className'),
            Output('button-right-engine', 'className'),
        ],
        [
            Input('tail-number', 'data'),
            Input('engine-number', 'data'),
            Input('airplane_table', 'active_cell'),
            Input('button-left-engine', 'n_clicks'),
            Input('button-right-engine', 'n_clicks'),
        ], prevent_initial_call=True)
def handle_plane_and_engine_selection(tail_number, engine_number, click_table, button_left_engine, button_right_engine):

    ctx = dash.callback_context
    if not ctx.triggered:
        print('No clicks yet')
    else:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if prop_id == "airplane_table":
            # we clicked in the table, so we change the tail_number
            row = click_table['row']
            tail_number = df_airplane[row:row+1]['tail_number'].tolist()[0]

        button_left_engine_className = "engine engine-left"
        button_right_engine_className = "engine engine-right"
        df = df_airplane[df_airplane.tail_number == tail_number]
        if len(df):
            left = df.left.tolist()[0]
            right = df.right.tolist()[0]
            if prop_id == "airplane_table":
                left_risk = df_test_final[df_test_final.unit_number == left].risk.tolist()[0]
                right_risk = df_test_final[df_test_final.unit_number == right].risk.tolist()[0]
                left_risk_class = risk_classes.index(left_risk)
                right_risk_class = ['low', 'medium', 'high'].index(right_risk)
                if left_risk_class >= right_risk_class:
                    engine_number = left
                else:
                    engine_number = right

            if prop_id == "button-left-engine":
                engine_number = left
            if prop_id == "button-right-engine":
                engine_number = right

            button_left_engine_className = "engine engine-left"
            button_right_engine_className = "engine engine-right"
            if engine_number == left:
                button_left_engine_className += " engine-active"
            if engine_number == right:
                button_right_engine_className += " engine-active"
    return tail_number, engine_number, button_left_engine_className, button_right_engine_className


@app.callback([Output('engine-info', 'children'),
               Output('shap-values', 'data')],
              Input('engine-number', 'data'),
              prevent_initial_call=True)
def update_engine_info(engine_number):
    engine_info = get_engine_info(engine_number=engine_number)
    if engine_number is not None:
        shap_values = get_shap_values(engine_number=engine_number)
    else:
        shap_values = []
    return engine_info, shap_values


@app.callback(Output('sensor-importance-figure', 'figure'),
              Input('shap-values', 'data'),
              prevent_initial_call=True)
def update_feature_importance_figure(shap_values):
    shap_values = np.array(shap_values)
    expl = compute_feature_importances(shap_values=shap_values)
    fig = create_feature_importance_figure(expl)
    return fig


@app.callback(Output('sensor-list', 'value'),
              Input('sensor-importance-figure', 'clickData'),
              prevent_initial_call=True)
def click_feature_importance(click_data):
    sensor = click_data['points'][0]['label']
    return sensor


@app.callback([
                Output('high-left', 'style'),
                Output('high-right', 'style'),
                Output('medium-left', 'style'),
                Output('medium-right', 'style')],
              Input('tail-number', 'data'),
              prevent_initial_call=True)
def show_airplane(tail_number):
    df = df_airplane[df_airplane.tail_number == tail_number]
    visible = {'display': 'unset'}
    invisible = {'display': 'none'}
    hl = hr = ml = mr = invisible
    if len(df):
        left = df.left.tolist()[0]
        right = df.right.tolist()[0]
        left_risk = df_test_final[df_test_final.unit_number == left].risk.tolist()[0]
        right_risk = df_test_final[df_test_final.unit_number == right].risk.tolist()[0]
    else:
        left_risk = right_risk = "low"
    if left_risk == "high":
        hl = visible
    if left_risk == "medium":
        ml = visible
    if right_risk == "high":
        hr = visible
    if right_risk == "medium":
        mr = visible
    return hl, hr, ml, mr


@app.callback(Output('explainer-figure', 'figure'),
              [Input('shap-values', 'data'),
               Input('sensor-list', 'value'),
               Input('scale', 'value')],
              State('engine-number', 'data'),
              prevent_initial_call=True)
def update_explain_sensor_figure(shap_values, sensor, scaled, engine_number):
    shap_values = np.array(shap_values)
    x, y, w = compute_sensor_data(engine_number=engine_number, sensor=sensor, shap_values=shap_values, scaled=scaled)
    fig = create_explain_sensor_figure(x=x, y=y, w=w, sensor=sensor)
    return fig


# #################################
# Run the dashboard
# #################################

if __name__ == '__main__':
    app.run_server(debug=True, host="0.0.0.0", port="8060")
