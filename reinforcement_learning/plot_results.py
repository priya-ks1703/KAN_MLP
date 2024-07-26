import json
from agent.utils import plot_model_performance

if __name__ == '__main__':
    # load total_results.json
    with open("./models_cartpole/total_results.json", "r") as f:
        total_results = json.load(f)

    # All plots
    plot_model_performance(total_results, dist_type='in_distribution', out_folder='./models_cartpole')
    plot_model_performance(total_results, dist_type='out_of_distribution', out_folder='./models_cartpole')

    # Pair plots
    plot_model_performance(total_results, dist_type='in_distribution', out_folder='./models_cartpole', model_types=['kan_9', 'mlp_40'], name='small')
    plot_model_performance(total_results, dist_type='in_distribution', out_folder='./models_cartpole', model_types=['kan_16_16', 'mlp_40_40'], name='medium')
    plot_model_performance(total_results, dist_type='in_distribution', out_folder='./models_cartpole', model_types=['kan_17_17_17', 'mlp_40_40_40'], name='large')

    plot_model_performance(total_results, dist_type='out_of_distribution', out_folder='./models_cartpole', model_types=['kan_9', 'mlp_40'], name='small')
    plot_model_performance(total_results, dist_type='out_of_distribution', out_folder='./models_cartpole', model_types=['kan_16_16', 'mlp_40_40'], name='medium')
    plot_model_performance(total_results, dist_type='out_of_distribution', out_folder='./models_cartpole', model_types=['kan_17_17_17', 'mlp_40_40_40'], name='large')