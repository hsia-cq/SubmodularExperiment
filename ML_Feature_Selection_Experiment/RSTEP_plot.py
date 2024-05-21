import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
K_values = np.arange(1, 8)  # Feature counts from 1 to 7

# Data for R=1 (7 values)
medians_R1 = [0.7244897959183674, 0.8010204081632653, 0.8290816326530612, 0.8724489795918368, 0.8954081632653061, 0.9081632653061225, 0.9183673469387755]
iqr_R1 = [0.10204081632653061, 0.030612244897959218, 0.020408163265306145, 0.029336734693877542, 0.02423469387755106, 0.010204081632653073, 0.010204081632653073]

# Data for R=2 (4 values at K=2, 3, 4, 5, 6, 7)
medians_R2 = [0.7806122448979592, 0.8367346938775511, 0.8724489795918368, 0.8979591836734694, 0.9132653061224489, 0.9183673469387755]
iqr_R2 = [0.02168367346938782, 0.015306122448979553, 0.02423469387755106, 0.015306122448979553, 0.014030612244897878, 0.010204081632653073]
K_values_R2 = [2, 3, 4, 5, 6, 7]

# Data for R=3 (3 values at K=3, 4, 5, 6, 7)
medians_R3 = [0.8418367346938775, 0.8775510204081632, 0.8979591836734694, 0.9132653061224489, 0.9209183673469388]
iqr_R3 = [0.015306122448979553, 0.014030612244897989, 0.017857142857142794, 0.010204081632653073, 0.010204081632653073]
K_values_R3 = [3, 4, 5, 6, 7]

# Plotting
fig, ax = plt.subplots(figsize=(18, 6))  # Adjust the overall figure size
bar_width = 0.28  # Adjust bar width to fill space

# Plotting R=1
ax.bar(K_values - bar_width, medians_R1, width=bar_width, color='#E89DA0', align='center', yerr=iqr_R1, label='RSOP-Greedy R=1', capsize=5, edgecolor='black', hatch='-')

# Plotting R=2, filling in values
medians_R2_full = [np.nan if k not in K_values_R2 else medians_R2[K_values_R2.index(k)] for k in K_values]
iqr_R2_full = [0 if k not in K_values_R2 else iqr_R2[K_values_R2.index(k)] for k in K_values]
ax.bar(K_values, medians_R2_full, width=bar_width, color='#88CEE6', align='center', yerr=iqr_R2_full, label='RSOP-Greedy R=2', capsize=5, edgecolor='black', hatch='/')

# Plotting R=3, filling in values
medians_R3_full = [np.nan if k not in K_values_R3 else medians_R3[K_values_R3.index(k)] for k in K_values]
iqr_R3_full = [0 if k not in K_values_R3 else iqr_R3[K_values_R3.index(k)] for k in K_values]
ax.bar(K_values + bar_width, medians_R3_full, width=bar_width, color='#D9B9D4', align='center', yerr=iqr_R3_full, label='RSOP-Greedy R=3', capsize=5, edgecolor='black', hatch='|')

# Setting labels, font sizes, and removing the title
ax.set_xlabel('Number of Features (K)', fontsize=30)
ax.set_ylabel('Accuracy', fontsize=30)
ax.tick_params(axis='y', labelsize=20)
ax.set_ylim(0.6, 1.0)
ax.set_xticks(K_values)
ax.set_xticklabels(K_values, fontsize=20)
ax.legend(fontsize=20, loc='upper left')
# Adding a dashed line at the top of the plot to indicate a boundary or threshold
ax.axhline(y=1.0, color='gray', linestyle='dashed')

plt.tight_layout()  # Adjust layout to make it more compact
# Save the figure as a PDF
plt.savefig(r'C:\Users\1\Desktop\r_airline.pdf', format='pdf', bbox_inches='tight')
plt.show()
