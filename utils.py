import json
from openai import AzureOpenAI
import os

# Azure OpenAI client
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    api_key=os.getenv("AZURE_API_KEY"),
    api_version="2024-02-01"
)


def extract_article(url):
    import newspaper
    article = newspaper.Article(url)
    article.download()
    article.parse()
    article.nlp()
    return article.title, article.summary, article.text

def get_sentiment(text):
    from textblob import TextBlob
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return "positive"
    elif polarity < -0.2:
        return "negative"
    else:
        return "neutral"

def extract_subtopics(text):
    system_prompt = """
You are a web story editor for a digital news platform.

Your task is to write 5 **slide titles** for a web story based on a news article. 
Each title represents a visual chapter in a 5-slide storytelling format.

Follow this logical structure:
1. Slide 1 — Hook: Lead with the core news moment or big headline.
2. Slide 2 — Who/What: Introduce the key person, backstory, or main subject.
3. Slide 3 — Inner Detail: Focus on something deeper—method, motive, insight, etc.
4. Slide 4 — Contrast or Conflict: Highlight a rivalry, twist, challenge, or critical moment.
5. Slide 5 — Future or Legacy: End with what it means for the future, impact, or reflection.

Return only the slide titles as a JSON list of 5 strings. Do NOT include explanations or extra text.
Keep titles short and punchy, ideally under 8 words.
"""

    user_prompt = f"""
Generate 5 slide titles for a news story based on this article:

\"\"\"
{text[:3000]}
\"\"\"

Return format:
[
  "Slide Title 1",
  "Slide Title 2",
  ...
]
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```json"):
        content = content.removeprefix("```json").strip()
    if content.endswith("```"):
        content = content.removesuffix("```").strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return [line.strip("- ").strip() for line in content.split("\n") if line.strip()]


def detect_category_and_subcategory(text):
    prompt = f"""
You are an expert news analyst.

Analyze the following news article and return:

1. The best matching **category** from this fixed list:
   - Politics
   - Business & Economy
   - Technology
   - Science & Environment
   - Health
   - Crime & Law
   - Sports
   - Entertainment
   - Lifestyle
   - Education
   - World / International
   - Local / Regional News
   - Opinion / Editorial
   - Religion & Spirituality
   - Obituaries & Tributes

2. A short, specific **subcategory** that summarizes the article’s main subject. 
   The subcategory should be highly relevant and can be:
   - a **person** (e.g., "Narendra Modi", "Gukesh Dommaraju")
   - an **event** (e.g., "Norway Chess 2025", "Budget 2024")
   - a **topic or issue** (e.g., "Mental Health in Schools", "Data Privacy")
   - a **location** (e.g., "Manipur", "Silicon Valley")
   - an **organization or institution** (e.g., "UNICEF", "Apple Inc.")
   - a **product or platform** (e.g., "ChatGPT", "Apple Vision Pro")
   - a **policy or law** (e.g., "Digital Personal Data Bill", "NEP 2020")
   - a **conflict or investigation** (e.g., "ED Probe", "Supreme Court Ruling")
   - a **cultural trend or movement** (e.g., "Remote Work Culture", "Clean Living")

3. The **primary emotion** evoked by the article (e.g., Pride, Sadness, Hope, Outrage, Empathy, Awe, Fear, Inspiration)

Article:
\"\"\"
{text[:3000]}
\"\"\"

Respond **only** in the following JSON format:
{{
  "category": "...",
  "subcategory": "...",
  "emotion": "..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You classify news articles by category, specific subcategory, and primary emotion evoked."},
            {"role": "user", "content": prompt.strip()}
        ]
    )

    content = response.choices[0].message.content.strip()

    if content.startswith("```json"):
        content = content.removeprefix("```json").strip()
    if content.endswith("```"):
        content = content.removesuffix("```").strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "category": "Unknown",
            "subcategory": "General",
            "emotion": "Neutral"
        }


def title_script_generator(category, subcategory, emotion, article_text, character_sketch=None):
    if not character_sketch:
        character_sketch = """
Rohan Sharma 35 sal ka ek naujawan hai jo ek bahut bade news channel mein news anchor ki naukri karta hai. Woh short hair aur clean shaved rahata hai. Khud Ko hamesha presentable rakhta hai. Rohan kafi intelligent aur sincere hai. Use bachpan Se hi current affairs aur storytelling pasand rahi hai. 
Bachpan mein usne apne papa se prerit hokar ek journalist banne ka faisla kiya tha. Usne journalism ki padhaai khatm karte hi field reporter ki naukari shuru kar di thi. Kam karte hue woh bahut sari jagahe ghuma aur zindagi ke bare mein aisi baatein jaani jo shayad kabhi na jaan pata. Uski saaf awaaz aur kisi bhi chij ko aasani se samjha dene wali quality ne use news anchor banne ke liye prerit kiya. Rohan hamesha har field mein up to date rahata hai aur puri koshish karta hai ki koi bhi galat khabar public tak na pahunche.
Use kitaben padhna pasand hai aur apne doston ke sath gehri baten karna. Wah kafi humble aur dil ka saaf insan hai aur uska maanana hai ki hamein hamesha ek behtar samaj banne ki or kadam uthana chahie.
"""

    # Step 1: Generate slide titles and narration prompts
    system_prompt = """
You are a digital content editor.

Create a structured 5-slide web story from the article below. Each slide must contain:
- A short English title (for the slide)
- A prompt: a clear instruction telling another GPT model what narration to write (don't write the narration here)

Format:
{
  "slides": [
    { "title": "...", "prompt": "..." },
    ...
  ]
}
"""
    user_prompt = f"""
Category: {category}
Subcategory: {subcategory}
Emotion: {emotion}

Article:
\"\"\"
{article_text[:3000]}
\"\"\"

Write 5 distinct slides with:
- A title (1 line, in English)
- A narration instruction (prompt) for that slide
Only return the JSON.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()}
        ]
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```json"):
        raw = raw.removeprefix("```json").strip()
    if raw.endswith("```"):
        raw = raw.removesuffix("```").strip()

    try:
        slides_raw = json.loads(raw)["slides"]
    except Exception:
        return {
            "category": category,
            "subcategory": subcategory,
            "emotion": emotion,
            "slides": []
        }

    # Step 2: Slide 1 — Short intro + Hindi-English headline
    headline = article_text.split("\n")[0].strip().replace('"', '').replace("’", "'")
    short_hinglish_headline_prompt = f"""
Convert the following news headline into a short Hindi-English mixed sentence suitable for narration:
"{headline}"
"""
    headline_response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": short_hinglish_headline_prompt.strip()}
        ]
    )
    short_hinglish_headline = headline_response.choices[0].message.content.strip()

    slide1_script = f"Namaskar doston, main hoon Rohan Sharma, aur aap dekh rahe hain Suvichaar Live. Aaj ki badi khabar: {short_hinglish_headline}"

    slides = [{
        "title": headline[:80],  # keep short title
        "prompt": "Start with an introduction by Rohan Sharma, greeting the audience and reading the headline.",
        "image_prompt": f"Create a clean vector-style illustration of Rohan Sharma introducing the news titled: '{headline}'.",
        "script": slide1_script
    }]

    # Step 3: Slides 2–6 — Short Hindi-English narration (no intro)
    for slide_data in slides_raw:
        title = slide_data.get("title", "").strip()
        prompt = slide_data.get("prompt", "").strip()

        narration_prompt = f"""
Write a short 3–4 line news narration in Hindi-English mix in the voice of Rohan Sharma, based on the instruction below.
Instruction: {prompt}

Metadata:
Category: {category}
Subcategory: {subcategory}
Emotion: {emotion}

Tone: Clear, warm, human, slightly formal. Use simple Hindi-English mix suitable for Indian digital news.

Avoid any introduction or sign-off. Do not repeat "Main hoon Rohan Sharma" etc.

Rohan’s style:
{character_sketch.strip()}
"""

        narration_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You write short Hindi-English narrations in the voice of Rohan Sharma."},
                {"role": "user", "content": narration_prompt.strip()}
            ]
        )
        script = narration_response.choices[0].message.content.strip()
        image_prompt = f"Generate a modern vector-style image for: '{title}'. Reflect themes of {category} / {subcategory}."

        slides.append({
            "title": title,
            "prompt": prompt,
            "image_prompt": image_prompt,
            "script": script
        })

    return {
        "category": category,
        "subcategory": subcategory,
        "emotion": emotion,
        "slides": slides
    }
