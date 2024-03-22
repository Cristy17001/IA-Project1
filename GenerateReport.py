def generate_report():
    # Import packages
    from dash import Dash, html, dash_table, dcc, callback, Output, Input
    import pandas as pd
    import plotly.express as px
    import json

    # Read the JSON file
    with open('report.json', 'r') as file:
        data = json.load(file)

    df_genetic = pd.DataFrame.from_dict(data['genetic'])
    if not(df_genetic.empty):
        df_genetic = df_genetic.round(2)
        df_genetic['min_solution'] = df_genetic['min_solution'].apply(lambda x: str(x))
        df_genetic = df_genetic.reset_index()
        df_genetic.rename(columns={'index': 'Generation', 'mean': 'Mean Fitness', 'min': 'Best Fitness', 'min_solution': 'Best Solution'}, inplace=True)

    df_simulated_annealing = pd.DataFrame.from_dict(data['simulated_annealing'])
    if not(df_simulated_annealing.empty):
        df_simulated_annealing = df_simulated_annealing.round(2)
        df_simulated_annealing['solutions'] = df_simulated_annealing['solutions'].apply(lambda x: str(x))
        df_simulated_annealing.rename(columns={'iterations': 'Iteration', 'fitness_scores': 'Best Fitness', 'solutions': 'Solution'}, inplace=True)

    df_first_airplane_stream = pd.DataFrame.from_dict(data['first_airplane_stream'])
    if not(df_first_airplane_stream.empty):
        df_first_airplane_stream = df_first_airplane_stream.round(2)
        df_first_airplane_stream.rename(columns={'arriving_fuel_level': 'Arriving Fuel Level', 'fuel_consumption_rate': 'Fuel Consumption Rate', 'expected_landing_time': 'Expected Landing Time'}, inplace=True)

    # Initialize the app - incorporate css
    external_stylesheets = ['style.css']

    # Initialize the app - incorporate css
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    # App layout
    app.layout = html.Div([
        # Title
        html.Div(className='title-wrapper', children=[
            html.Div(className='title', children='Optimization Problems - Airport Landing'),
        ]),

        html.Div(className='select-table',children=[
            html.Div(className='row', children=[
                dcc.Dropdown(options=[{'label': 'Genetic Algorithm', 'value': 'genetic'},
                                      {'label': 'Simulated annealing', 'value': 'simulated_annealing'}],
                             value='genetic',
                             id='my-dropdown-final'),
            ]),
            html.Div(id='table',children=[]),
        ]),

        html.Div(className='row', children=[
            html.Div(children=[
                dcc.Graph(figure={}, id='line-chart-final1')
            ]),
            html.Div(children=[
                dcc.Graph(figure={}, id='line-chart-final2')
            ])
        ]),

        html.Div(className='airplane_stream', children=[
            html.Div(className='airplane_stream_title', children='First Airplane Stream'),
        ]),
        html.Div(className='initial_airplane_stream', id='first_airplane_stream', children=[
            dash_table.DataTable(data=df_first_airplane_stream.to_dict('records'), page_size=15),]),
    ])

    # Add controls to build the interaction
    @callback(
        [Output(component_id='line-chart-final1', component_property='figure'),
         Output(component_id='line-chart-final2', component_property='figure'),
         Output(component_id='table', component_property='children')],
        Input(component_id='my-dropdown-final', component_property='value')
    )
    def update_graph(option_chosen):
        if option_chosen == 'genetic':
            if not(df_genetic.empty):
                table = dash_table.DataTable(data=df_genetic.to_dict('records'), page_size=15)
                fig1 = px.line(df_genetic, x='Generation', y='Mean Fitness', template='plotly_white')
                fig1.update_layout(
                    title="Mean Fitness over Generations",
                    xaxis_title="Generation",
                    yaxis_title="Mean Fitness"
                )
                fig2 = px.line(df_genetic, x='Generation', y='Best Fitness', template='plotly_white')
                fig2.update_layout(
                    title="Minimum Fitness over Generations",
                    xaxis_title="Generation",
                    yaxis_title="Minimum Fitness"
                )
            else:
                fig1 = px.line([], x=[0], y=[0], template='plotly_white', title='No data available for this algorithm')
                fig2 = px.line([], x=[0], y=[0], template='plotly_white', title='No data available for this algorithm')
                table = None

        elif option_chosen == 'simulated_annealing':
            if not(df_simulated_annealing.empty):
                table = dash_table.DataTable(data=df_simulated_annealing.to_dict('records'), page_size=15)
                fig1 = px.line(df_simulated_annealing, x='Iteration', y='Best Fitness', template='plotly_white')
                fig1.update_layout(
                    title="Fitness over Iterations",
                    xaxis_title="Iteration",
                    yaxis_title="Mean Fitness"
                )
                fig2 = px.line([], x=[0], y=[0], template='plotly_white', title='No data available for this algorithm')
            else:
                fig1 = px.line([], x=[0], y=[0], template='plotly_white', title='No data available for this algorithm')
                fig2 = px.line([], x=[0], y=[0], template='plotly_white', title='No data available for this algorithm')
                table = None


        return fig1, fig2, table

    # Run the app
    # if __name__ == '__main__':
    app.run(debug=True)

# Call the function to generate the report
generate_report()
