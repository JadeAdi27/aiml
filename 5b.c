import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

# Load dataset and prepare data
seeds_df = pd.read_csv('seeds-less-rows.csv')
varieties = list(seeds_df.pop('grain_variety'))
samples = seeds_df.values

# Hierarchical clustering
mergings = linkage(samples, method='complete')
dendrogram(mergings, labels=varieties, leaf_rotation=90, leaf_font_size=6)
plt.show()

# Extract cluster labels
labels = fcluster(mergings, 6, criterion='distance')

# Cross-tabulation of cluster labels and grain varieties
df = pd.DataFrame({'labels': labels, 'varieties': varieties})
ct = pd.crosstab(df['labels'], df['varieties'])
print(ct)
