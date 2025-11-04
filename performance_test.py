import requests
import time
import csv

URL = "http://ece444pra5-env.eba-ve523ppe.us-east-2.elasticbeanstalk.com/predict"
TEST_CASES = {
    "fake1": "India is a city.",
    "fake2": "There are 10 continents in the world.",
    "real1": "New York City is a city.",
    "real2": "New York City is located in the United States."
}

def run_test(case_name, text):
    filename = f"{case_name}.csv"
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["request_number", "latency_seconds"])

        for i in range(100):
            start = time.time()
            r = requests.post(URL, json={"message": text})
            latency = time.time() - start
            writer.writerow([i+1, latency])
            print(case_name, i+1, latency, "response:", r.json())
    print(f"Saved: {filename}")

for case, text in TEST_CASES.items():
    run_test(case, text)

import pandas as pd
import glob

# After running all tests, calculate and print averages
files = glob.glob("*.csv")
print("\nAverage Latency per Test Case:")
for file in files:
    df = pd.read_csv(file)
    avg_latency = df["latency_seconds"].mean()
    print(f"{file.replace('.csv', '')}: {avg_latency:.4f} seconds")

