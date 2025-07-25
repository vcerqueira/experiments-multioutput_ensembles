import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import plotnine as p9

past_length = 3000
future_length = 6000

# Generate a single past path
past_rw = np.cumsum(np.random.choice([-1, 1], size=past_length))
past_df = pd.DataFrame({'index': np.arange(past_length), 'value': past_rw})

# The last value of the past is the starting point for all future paths
start_value = past_rw[-1]

# Generate 5 future random walks to simulate different model forecasts
n_models = 5
model_names = [f'Model {i+1}' for i in range(n_models)]
future_rws = np.array([
    start_value + np.cumsum(np.random.choice([-1, 1], size=future_length))
    for _ in range(n_models)
])

# Calculate the true value as the average of forecasts
true_value = np.mean(future_rws, axis=0)
true_df = pd.DataFrame({
    'index': np.arange(past_length, past_length + future_length),
    'value': true_value,
    'Model': 'True value'
})

future_df = pd.DataFrame(future_rws).T
future_df['index'] = np.arange(past_length, past_length + future_length)
future_df_reset = future_df.melt(id_vars='index', var_name='Model', value_name='value')
# Assign model names instead of numbers
future_df_reset['Model'] = future_df_reset['Model'].map(lambda x: model_names[x])

# Combine all forecasts and true value
all_forecasts = pd.concat([future_df_reset, true_df])

# Create a DataFrame for the vertical line annotation
vline_text = pd.DataFrame({
    'x': [past_length+400],
    'y': [np.max(past_rw) + (np.max(future_rws) - np.min(future_rws)) * 0.3],  # Position text closer to the lines
    'label': ['Forecast\norigin']
})

# Define a color palette for the models
model_colors = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6"]

rw_plot_combined = \
    p9.ggplot() + \
    p9.geom_line(data=past_df,
                 mapping=p9.aes(x='index', y='value'),
                 color='black',
                 size=1.2) + \
    p9.geom_line(
        data=all_forecasts,
        mapping=p9.aes(x='index', y='value', group='Model',
                      color='Model', linetype='Model'),
        size=1,
        alpha=0.7
    ) + \
    p9.geom_vline(xintercept=past_length, linetype='dotted', color='gray', size=1) + \
    p9.geom_text(
        data=vline_text,
        mapping=p9.aes(x='x', y='y', label='label'),
        ha='center',
        size=10,
        family='Palatino'
    ) + \
    p9.scale_color_manual(values=model_colors + ['black']) + \
    p9.scale_linetype_manual(values=['solid']*5 + ['dashed']) + \
    p9.theme_classic(base_family='Palatino', base_size=12) + \
    p9.theme(
        plot_margin=.125,
        axis_text=p9.element_blank(),
        axis_ticks_major=p9.element_blank(),
        axis_ticks_minor=p9.element_blank(),
        axis_line=p9.element_blank(),
        legend_position='right',
        legend_background=p9.element_rect(fill='white', alpha=0.8)
    ) + \
    p9.labs(x='', y='')

rw_plot_combined.save('example_multistep_forecasts.pdf', width=12, height=7)
