import speech_recognition as sr
import pyttsx3
from googletrans import Translator, LANGUAGES
translator = Translator()
engine = pyttsx3.init()
engine.setProperty('rate', 150)
def speak(text):
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"❌ Speech Error: {e}")
def display_languages():
    print("\n🌍 Available Languages:\n")
    language_codes = list(LANGUAGES.keys())
    for index, code in enumerate(language_codes, start=1):
        print(f"{index}. {LANGUAGES[code].title()} ({code})")
    return language_codes
def get_language_choice(language_codes, purpose="source"):
    while True:
        try:
            choice = int(
                input(f"\nSelect {purpose} language number: ")
            )
            if 1 <= choice <= len(language_codes):
                selected_code = language_codes[choice - 1]
                selected_name = LANGUAGES[selected_code].title()
                print(f"✅ Selected {purpose} language: "
                      f"{selected_name} ({selected_code})")
                return selected_code
            else:
                print("❌ Invalid choice. Try again.")
        except ValueError:
            print("❌ Please enter a valid number.")
def speech_to_text(source_language):
    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("\n🎤 Adjusting for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🎙️ Speak now...")
            audio = recognizer.listen(source, timeout=10)
        print("🔍 Recognizing speech...")
        text = recognizer.recognize_google(
            audio,
            language=source_language
        )
        print(f"✅ Recognized Text: {text}")
        return text
    except sr.WaitTimeoutError:
        print("❌ No speech detected within time limit.")
    except sr.UnknownValueError:
        print("❌ Could not understand the speech clearly.")
    except sr.RequestError as e:
        print(f"❌ Speech Recognition API Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
    return ""
def translate_text(text, target_language):
    try:
        translation = translator.translate(
            text,
            dest=target_language
        )
        print(f"\n🌐 Translated Text: {translation.text}")
        return translation.text
    except Exception as e:
        print(f"❌ Translation Error: {e}")
        return ""
def main():
    print("===================================")
    print("🎧 Real-Time Speech Translation App")
    print("===================================")
    language_codes = display_languages()
    source_language = get_language_choice(
        language_codes,
        "source"
    )
    target_language = get_language_choice(
        language_codes,
        "target"
    )
    original_text = speech_to_text(source_language)
    if original_text:
        translated_text = translate_text(
            original_text,
            target_language
        )
        if translated_text:
            print("🔊 Speaking translated text...")
            speak(translated_text)
            print("✅ Translation completed successfully!")
        else:
            print("❌ Translation failed.")
    else:
        print("❌ Speech recognition failed.")
if __name__ == "__main__":
    main()