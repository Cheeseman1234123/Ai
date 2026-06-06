from groq import generate_response
def reinforcement_learning_activity():
    print("\n=== REINFORCEMENT LEARNING ACTIVITY ===\n")
    prompt = input("Enter a prompt for the AI model (e.g. 'Describe the lion'):").strip()
    if not prompt:
        print("Please enter a prompt to run this activity.")
        return
    initial_response = generate_response(prompt, temperature=0.3, max_tokens=3)
    print(f"\nInitial AI response: {initial_response}")
    try:
        rating = int(input("Rate the response from 1 (bad) to 5 (good): ").strip())
        if rating < 1 or rating > 5:
            print("Invalid rating. Using 3")
            rating = 3
    except ValueError:
        print("Invalid rating. Using 3")
        rating = 3
    feedback = input("Provide feedback for improvement: ").strip()
    improved_response = f"{initial_response} (Improve with your feedback: {feedback})"
    print(f"\nImproved AI response : {improved_response}")
    print("\nReflection:")
    print("1. How did the model's response improve with your feedback?")
    print("2. How does reinforcement learning help AI to improve its performance over time?")
def role_based_prompt_activity():
    print("\n=== Role Based Prompt Activity ===\n")
    category = input("Enter a category (e.g., science, history, math): ").strip()
    item = input(f"Enter a specific {category} topic (e.g., 'photosynthesis' for science)").strip()
    if not category or not item:
        print("Please fill in both fields to run this activity")
        return
    teacher_response = f"You are a teacher. Explain {item} in simple terms."
    expert_response = f"You are an expert in {category}. Explain {item} in a detailed, technical manner."
    print(f"\n--- Teacher's Perspective ---\n{teacher_response}")
    print(f"\n--- Expert's Perspective ---\n{expert_response}")
    print("\nReflection:")
    print("1. How did the Ai's response differ between the teacher's rperspective and the expert's perspective?")
    print("2. How can role based prompts help tailor AI responses for different contexts?")
def run_activity():
    print("\n=== AI Learning Activity ===")
    print("Choose an activity")
    print("1) Reinforcement Learning")
    print("2) Role-Based Prompts")
    choice = input("> ").strip()
    if choice == "1":
        reinforcement_learning_activity()
    elif choice == "2":
        role_based_prompt_activity()
    else:
        print("Invalid choice. Please choose 1 or 2.")
if __name__ == "__main__":
    run_activity()