import matplotlib.pyplot as plt
import numpy as np

# Define methods and their respective data
methods = ['Coverage', 'SOP-Greedy Coverage', 'Diversity', 'SOP-Greedy Diversity', 'Facility Location', 'SOP-Greedy Facility Location']
k_values = [3, 4, 5, 6, 7]

# Colors and hatches for aesthetics
colors = ['#D9F5F4', '#00C1C2', '#E9E6F7', '#6A5DC4', '#FDE9BD', '#FCD57A']
hatches = ['/', '\\', '|', '-', '+', 'x']

# Median and IQR data for each K-value
medians = {
    3: [0.6593, 0.5298, 0.3897, 0.3840, 0.5138, 0.4219],
    4: [0.7150, 0.5928, 0.4536, 0.4359, 0.5537, 0.4860],
    5: [0.7492, 0.6124, 0.5279, 0.4912, 0.5544, 0.5348],
    6: [0.7598, 0.6457, 0.5459, 0.5337, 0.5607, 0.5655],
    7: [0.7431, 0.6634, 0.5613, 0.5732, 0.5727, 0.5893]
}
iqrs = {
    3: [0.2666, 0.2586, 0.2152, 0.2345, 0.2297, 0.2332],
    4: [0.2347, 0.2314, 0.2694, 0.2432, 0.1937, 0.2375],
    5: [0.2586, 0.2042, 0.1922, 0.2296, 0.2024, 0.2274],
    6: [0.1783, 0.1865, 0.2191, 0.2194, 0.1805, 0.2051],
    7: [0.1545, 0.1637, 0.2041, 0.2098, 0.1850, 0.1974]
}

# Setting up the figure and axes
fig, ax = plt.subplots(figsize=(36, 5.5))
group_width = 0.87 # Total width for groups
bar_width = group_width / (len(methods) +2) # Individual bar width

# Generate positions for each group based on K-values
x_base = np.arange(len(k_values))

# Plotting
for k_idx, k in enumerate(k_values):
    offsets = x_base[k_idx] + np.linspace(0, group_width - bar_width, len(methods))
    medians_k = medians[k]
    iqrs_k = iqrs[k]
    for m_idx, method in enumerate(methods):
        ax.bar(offsets[m_idx], medians_k[m_idx], width=bar_width, color=colors[m_idx],
               yerr=iqrs_k[m_idx], capsize=5, label=method if k_idx == 0 else "", hatch=hatches[m_idx],
               edgecolor='black')
        # Adding median labels
        ax.text(offsets[m_idx], medians_k[m_idx] + 0.01, f'{medians_k[m_idx]:.2f}',
                ha='center', va='bottom', fontsize=19)

# Customizing the plot
ax.set_xticks(x_base + group_width / 2 - bar_width / 2)
ax.set_xticklabels([f'K={k}' for k in k_values], fontsize=26)
ax.set_ylabel('ROUGE-1 Score F-Measures', fontsize=25)
ax.tick_params(axis='y', labelsize=26)
ax.set_ylim(0.3, 1)
ax.axhline(y=1.0, color='gray', linestyle='dashed', linewidth=1, dash_capstyle='round', dashes=(5, 10))
ax.legend(loc='upper right', fontsize=17.2)

plt.tight_layout()
plt.savefig(r'C:\Users\1\Desktop\news_values.pdf', format='pdf', bbox_inches='tight')
plt.show()
