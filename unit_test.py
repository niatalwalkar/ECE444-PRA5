import requests

URL = "http://ece444pra5-env.eba-ve523ppe.us-east-2.elasticbeanstalk.com/predict"


TEST_CASES = {
    "fake1": "India is a city.",
    "fake2": "There are 10 continents in the world.",
    "real1": "New York City is a city.",
    "real2": "New York City is located in the United States."
}

for name, text in TEST_CASES.items():
    response = requests.post(URL, json={"message": text})
    print(f"{name}: {response.status_code} -> {response.json()}")
