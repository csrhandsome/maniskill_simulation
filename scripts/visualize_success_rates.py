import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use seaborn style for clean aesthetics
sns.set_theme(style='whitegrid', context='talk')

# Optional: LaTeX-style fonts (requires LaTeX installed)
plt.rcParams.update({
    'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],
    'text.usetex': False,  # Set to True if LaTeX is available
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12
})

# open door
# du: 0.86, 0.84, 0.84, 0.92, 0.84, 0.82, 0.84, 0.8, 0.9, 0.88, 0.86, 0.88, 0.78, 0.84, 0.8, 0.86, 0.88, 0.76, 0.96, 0.86                                                                                           
# idu: 0.84, 0.92, 0.88, 0.9, 0.88, 0.88, 0.82, 0.88, 0.9, 0.84, 0.9, 0.8, 0.92, 0.9, 0.9, 0.88, 0.92, 0.84, 0.92, 0.92                                                                                         

# Data
epochs = list(range(50, 1001, 50))
success_rates_model1 = [0.86, 0.84, 0.84, 0.92, 0.84, 0.82, 0.84, 0.8, 0.9, 0.88, 0.86, 0.88, 0.78, 0.84, 0.8, 0.86, 0.88, 0.76, 0.96, 0.86]
success_rates_model2 = [0.84, 0.92, 0.88, 0.9, 0.88, 0.88, 0.82, 0.88, 0.9, 0.84, 0.9, 0.8, 0.92, 0.9, 0.9, 0.88, 0.92, 0.84, 0.92, 0.92]

# top_10 = sorted(success_rates_model2[:12], reverse=True)[:10]
# # Compute the mean
# mean_top_10 = np.mean(top_10)
# print("Mean of top 10 values:", mean_top_10)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, success_rates_model1, label='Model 1', color='tab:blue', marker='o', markersize=6, linewidth=2.5)
plt.plot(epochs, success_rates_model2, label='Model 2', color='tab:orange', marker='s', markersize=6, linewidth=2.5)

# Labels and title
plt.xlabel('Epoch')
plt.ylabel('Success Rate')
plt.title('Success Rate vs. Epochs')

# Grid with dashed lines
plt.grid(True, linestyle='--', linewidth=1, alpha=0.7)

# Legend
plt.legend(loc='lower right')

# Layout
plt.tight_layout()

# Optional: Save figure
# plt.savefig("success_rate_plot.pdf", format='pdf', bbox_inches='tight')

plt.show()
