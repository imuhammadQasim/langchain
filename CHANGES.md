# main.py mein kya changes hue — tafseel

## Asal masla (root cause)
Aap ka `.env` file mein sirf `GROQ_API_KEY` tha, lekin code `openai:gpt-5.5` model
use kar raha tha (OpenAI ke liye). Na to OpenAI ki key thi, na `.env` file load ho
raha tha, na `langchain-openai` package installed thi. Isi wajah se code chal hi
nahi sakta tha.

---

## 1. `.env` file load nahi ho rahi thi

**Pehle:**
```python
from langchain.agents import create_agent
```

**Ab:**
```python
import sys
from dotenv import load_dotenv
from langchain.agents import create_agent

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()
```

**Kyun:** `load_dotenv()` call kiye baghair Python ko pata hi nahi chalta k
`.env` file mein `GROQ_API_KEY` maujood hai. Ye line file ko read karke us
key ko environment variable ke tor par set kar deti hai.

---

## 2. Ghalat provider/model — OpenAI ki jagah Groq

**Pehle:**
```python
model="openai:gpt-5.5",
```

**Ab:**
```python
model="groq:openai/gpt-oss-120b",
```

**Kyun:** Aap ke paas OpenAI ki API key hi nahi hai — sirf Groq ki hai. Isliye
provider `"groq"` hona chahiye, `"openai"` nahi. `gpt-5.5` bhi asal mein
OpenAI ka koi valid model nahi hai.

Maine aap ki Groq account se live query kar ke check kiya k abhi kaun se
models available hain (kuch purane models jaise `llama-3.3-70b-versatile`
ab deprecated ho chuke hain). `openai/gpt-oss-120b` available hai aur
tool-calling support karta hai (yaani ye `get_weather` function ko call
kar sakta hai) — isliye ye choose kiya.

---

## 3. Windows terminal encoding crash

Jab model ne reply diya to us mein ek emoji (🌞ish character) tha, aur
Windows ka default terminal encoding (`cp1252`) us emoji ko print nahi kar
saka — is wajah se `UnicodeEncodeError` aata tha.

**Fix:**
```python
sys.stdout.reconfigure(encoding="utf-8")
```

Ye line terminal ko UTF-8 mein print karne par majboor karti hai, taake
koi bhi emoji ya special character crash na kare.

---

## 4. `requirements.txt` mein invalid syntax

**Pehle:**
```
langchain=1.3.16
```

**Ab:**
```
langchain==1.3.16
langchain-groq
python-dotenv
```

**Kyun:** pip mein version specify karne ke liye `==` chahiye hota hai,
sirf `=` se error aata hai. Saath hi `langchain-groq` (Groq ke sath baat
karne ke liye) aur `python-dotenv` (`.env` file read karne ke liye) missing
thi — dono add ki.

---

## Result

Code ab successfully chal raha hai. Agent `get_weather` tool ko call karta
hai aur uske jawab ke base par ek natural-language response deta hai:

```
San Francisco is currently sunny. Enjoy the clear skies!
```

---

## Agla step (agar zaroorat ho)
Agar aap actually **OpenAI** use karna chahte hain (Groq nahi), to bataiye —
mein:
- `model` wapis `openai:...` par set kar dunga (koi valid model, jaise `gpt-4o-mini`),
- `requirements.txt` mein `langchain-openai` add kar dunga,
- aur aap ko bata dunga k `.env` mein `OPENAI_API_KEY=...` add karni hai.
