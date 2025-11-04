import pandas as pd
import matplotlib.pyplot as plt
import glob

files = glob.glob("*.csv")
latencies = {}

for file in files:
    df = pd.read_csv(file)
    case = file.replace(".csv", "")
    latencies[case] = df["latency_seconds"]

# Boxplot
plt.figure(figsize=(8,5))
plt.boxplot(latencies.values(), labels=latencies.keys())
plt.title("API Latency Comparison (100 calls each)")
plt.ylabel("Latency (seconds)")
plt.xlabel("Test Case")
plt.show()

# Print averages
for case, values in latencies.items():
    print(f"{case} average latency: {values.mean():.4f}s")
