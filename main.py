import requests
general_url = "https://uselessfacts.jsph.pl/random.json?language=en"
technology_url = "https://uselessfacts.jsph.pl/category/Technology.json?language=en"
history_url = "https://uselessfacts.jsph.pl/category/History.json?language=en"
science_url = "https://uselessfacts.jsph.pl/category/Science.json?language=en"
def get_fact(url, label):
    response = requests.get(url)
    if response.status_code == 200:
        fact_data = response.json()
        print(label + ": " + fact_data['text'])
    else:
        print("Failed to fetch text")
while True:
    print("\nChoose an option:")
    print("1. General Fact")
    print("2. Technology Fact")
    print("3. History Fact")
    print("4. Science Fact")
    print("q. Quit")
    user_input = input("Enter your choice:")
    if user_input.lower() == "q":
        break
    elif user_input == "1":
        get_fact(general_url, "General Fact")
    elif user_input == "2":
        get_fact(technology_url, "Technology Fact")
    elif user_input == "3":
        get_fact(history_url, "History Fact")
    elif user_input == "4":
        get_fact(science_url, "Science Fact")
    else:
        print("Invalid Choice")