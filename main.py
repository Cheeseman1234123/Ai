from config import HF_API_KEY
import requests, base64, os, re, time
from PIL import Image
from colorama import init, Fore, Style
init(autoreset=true)
ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type": "application/json"}

VISION_MODELS = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct"
]

TEXT_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    "mistralai/Mixtral-8x7B-Instruct-v0.1"
]
def _data_url(path: str) -> str:
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
def query_hf_api(payload: dict):
    try:
        r = requests.post(ROUTER_URL, headers=HEADERS, json=payload, timeout=120)
    except requests.RequestException as e:
        return None, f"Request failed: {e}"
    if r.status_code != 200:
        try:
            j = r.json()
            msg = j.get("error")
        except Exception:
            msg = (r.text or "").strip() or r.reason or "Request failed."
        return None, f"Status {r.status_code}: {msg}"
    try:
        return r.json(), None
    except Exception:
        return None, "Non-JSON response recieved from the API"
def _extract_text(data) -> str:
    msg = (data or {}).get("choices", [{}])[0].get("message", []) or {}
    return (msg.get("content") or "").strip()
def _run_models(models, messages, max_tokens=160, temperature=0.3):
    last_err = None
    for model in models:
        data, err = query_hf_api({"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature})
        if err:
            last_err = err
            continue
        out = _extract_text(data)
        if out:
            return out, None
        last_err = "Empty response from model"
    return None, last_err or "All models failed."
def words(text: str):
    return re.findall(r"\S+", (text or "").strip())
def _exact_n_words(text: str, n: int) -> str:
    return " ".join(_words(text)[:n])
def _ensure_sentence_end(text: str, n: int) -> str:
    t = (text or "").strip()
    if t and t[-1] not in ".!?":
        t += "."
    return t
def generate_text(prompt: str, max_new_tokens: int = 220) -> str:
    msgs = [{"role": "user", "content": prompt}]
    out, err = _run_models(TEXT_MODELS, msgs, max_tokens=max_new_tokens)
    if err:
        return f"[Error] {err}"
    return out
def generate_exact_sentence(prompt: str, n_words: int, max_new_tokens: int, tries: int = 6) -> str:
    for i in range(tries):
        text = generate_text(prompt, max_new_tokens)
        if text.startswith("[Error]"):
            continue
        words = _words(text)
        if len(words) >= n_words:
            return _ensure_sentence_end(_exact_n_words(text, n_words))
    return "Failed to match word count after several tries"
def get_basic_caption(image_path: str) -> str:
    print(f"{Fore.YELLOW}🖼️  Generating basic caption ...") 
    msgs = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "write one complete sentence describing this image in detail."},
            {"type": "image_url", "image_url": {"url:" _data_url(image_path)}}
        ]
    }]
    cap, err = 