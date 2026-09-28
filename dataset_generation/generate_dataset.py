import os
import re
import json
import time
from dotenv import load_dotenv
from google import genai
from groq import Groq

load_dotenv()

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

OUTPUT_PATH = "data/synthetic/devops_synthetic.jsonl"

# topic -> (count, model_source)
CATEGORIES = {
    "Git/GitHub": (60, "groq"),
}

PROMPT_TEMPLATE = """
You are an expert DevOps engineer.

Generate exactly {count} high-quality instruction-response pairs
for a DevOps AI assistant called "DevOps Sentinel".

Topic: {topic}

Requirements:
- Mix question types: troubleshooting, "explain this concept", "what does this command do", and "compare X vs Y".
- Each instruction should be a realistic DevOps question or task.
- Each response must be technically accurate.
- Include practical commands where appropriate.
- Avoid duplicate or near-duplicate questions.
- Keep responses concise but useful.
- Return ONLY valid JSON, no markdown.

Format:
[
    {{"instruction": "question", "response": "answer"}}
]
"""


def fix_invalid_escapes(text: str) -> str:
    return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', text)


def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()
    return fix_invalid_escapes(text)




def call_gemini(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(max_retries):
        try:
            response = gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Gemini error, retrying in 10s (attempt {attempt+1}/{max_retries})...")
                time.sleep(10)
            else:
                raise


def call_groq(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            wait_time = 20 if "429" in str(e) or "rate_limit" in str(e).lower() else 8
            if attempt < max_retries - 1:
                print(f"  Groq error ({type(e).__name__}). Waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait_time)
            else:
                raise


def generate_for_topic(topic: str, total_count: int, source: str, max_retries: int = 3):
    all_pairs = []
    chunk_size = 10
    
    while len(all_pairs) < total_count:
        count = min(chunk_size, total_count - len(all_pairs))
        prompt = PROMPT_TEMPLATE.format(count=count, topic=topic)

        for attempt in range(max_retries):
            try:
                if source == "gemini":
                    raw_text = call_gemini(prompt)
                elif source == "groq":
                    raw_text = call_groq(prompt)
                else:
                    raise ValueError(f"Unknown source: {source}")

                text = clean_json_text(raw_text)
                pairs = json.loads(text)
                
                for item in pairs:
                    item["category"] = topic
                    item["source_model"] = source
                
                all_pairs.extend(pairs)
                time.sleep(15)   # <-- ye line add karo, cooldown between successful chunks
                break # Success for this chunk
                
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    print(f"  JSONDecodeError ({e}). Retrying chunk (attempt {attempt+1}/{max_retries})...")
                    time.sleep(2)
                else:
                    print(f"  Failed to generate a valid chunk after {max_retries} retries. Skipping remaining for this topic.")
                    return all_pairs
    return all_pairs


def load_existing_instructions(path: str) -> set:
    seen = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    seen.add(json.loads(line)["instruction"].strip().lower())
    return seen


def main():
    existing = load_existing_instructions(OUTPUT_PATH)
    total_written = 0

    with open(OUTPUT_PATH, "a", encoding="utf-8") as file:
        for topic, (count, source) in CATEGORIES.items():
            print(f"Generating {count} pairs for: {topic} (via {source})")

            try:
                pairs = generate_for_topic(topic, count, source)
            except Exception as e:
                print(f"  FAILED for {topic}, skipping this category: {e}")
                continue

            for item in pairs:
                key = item["instruction"].strip().lower()
                if key in existing:
                    print(f"  Skipped duplicate: {item['instruction'][:60]}...")
                    continue
                existing.add(key)
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_written += 1

    print(f"\nDone. Added {total_written} new pairs to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()