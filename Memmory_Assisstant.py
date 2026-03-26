"""
Shopping Memory Assistant  (v5 — Live Price APIs)
Powered by Groq API  |  Multi-turn memory  |  Live prices from:
  • data.gov.in  Agmarknet API  → vegetables / fruits / grains
  • Open Prices API (Open Food Facts) → packaged / branded items
  • Static catalog fallback if APIs are unavailable
"""

import json
import re
import time
import requests
from functools import lru_cache
from openai import OpenAI

# ── Groq client ────────────────────────────────────────────────────────────────
client = OpenAI(
    api_key="Groq_Api",
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "openai/gpt-oss-20b"

# ══════════════════════════════════════════════════════════════════════════════
# API CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# ─── data.gov.in  ─────────────────────────────────────────────────────────────
# Register free at https://data.gov.in to get your API key
# Dataset: Agmarknet daily mandi prices (vegetables, fruits, grains)
DATA_GOV_API_KEY = "579b464db66ec23bdd000001cdd3946e44ce4aad38d07d1a3f026003"  # demo key — replace with yours
DATA_GOV_RESOURCE_ID = "9ef84268-d588-465a-a308-a864a43d0070"  # Agmarknet daily prices
DATA_GOV_BASE_URL = "https://api.data.gov.in/resource/{resource_id}"

# ─── Open Prices (Open Food Facts) ────────────────────────────────────────────
# Completely free, no key required
OPEN_PRICES_BASE_URL = "https://prices.openfoodfacts.org/api/v1"

# ── Simple in-process cache  (key → (price, timestamp)) ───────────────────────
_price_cache: dict[str, tuple[float, float]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour

# ══════════════════════════════════════════════════════════════════════════════
# STATIC FALLBACK CATALOG
# (used when both APIs fail or item is not found)
# ══════════════════════════════════════════════════════════════════════════════

BRANDED_CATALOG: dict[str, dict] = {
    "pen":          {"brands": ["Reynolds","Cello","Parker","Pilot"],          "colors": ["black","blue","red","green"],          "sizes": None,                                          "avg_price": 20,   "unit": "piece"},
    "notebook":     {"brands": ["Classmate","Navneet","Oxford","Spirax"],       "colors": ["blue","red","green","yellow","white"],  "sizes": ["A4","A5","B5"],                              "avg_price": 80,   "unit": "piece"},
    "pencil":       {"brands": ["Apsara","Nataraj","Staedtler","Faber-Castell"],"colors": ["natural wood","black"],                "sizes": ["HB","2B","4B"],                              "avg_price": 10,   "unit": "piece"},
    "eraser":       {"brands": ["Apsara","Camlin","Faber-Castell"],             "colors": ["white","pink","blue"],                 "sizes": None,                                          "avg_price": 5,    "unit": "piece"},
    "sharpener":    {"brands": ["Apsara","Camlin","Maped"],                     "colors": ["blue","green","red","yellow"],         "sizes": None,                                          "avg_price": 10,   "unit": "piece"},
    "ruler":        {"brands": ["Camlin","Classmate","Staedtler"],              "colors": ["transparent","white","yellow"],        "sizes": ["15cm","30cm","45cm","60cm"],                  "avg_price": 25,   "unit": "piece"},
    "stapler":      {"brands": ["Kangaro","Rapid","MAX"],                       "colors": ["black","grey","blue","red"],           "sizes": ["mini","standard","heavy-duty"],               "avg_price": 150,  "unit": "piece"},
    "scissors":     {"brands": ["Maped","Fiskars","Camlin"],                    "colors": ["red/grey","blue/grey","orange/grey"],  "sizes": ["small","medium","large"],                    "avg_price": 80,   "unit": "piece"},
    "glue stick":   {"brands": ["Fevistik","UHU","Scotch"],                     "colors": ["white (dries clear)"],                "sizes": ["8g","15g","21g","40g"],                      "avg_price": 30,   "unit": "piece"},
    "highlighter":  {"brands": ["Stabilo","Camlin","Luxor"],                    "colors": ["yellow","pink","green","orange","blue"],"sizes": None,                                         "avg_price": 40,   "unit": "piece"},
    "marker":       {"brands": ["Cello","Camlin","Edding","Staedtler"],         "colors": ["black","blue","red","green","white"],  "sizes": ["fine","medium","broad"],                     "avg_price": 35,   "unit": "piece"},
    "calculator":   {"brands": ["Casio","Sharp","Orpat","Texas Instruments"],   "colors": ["black","white","grey"],               "sizes": ["basic","scientific","graphing"],              "avg_price": 400,  "unit": "piece"},
    "bag":          {"brands": ["Wildcraft","Skybags","American Tourister","Nike"],"colors":["black","blue","red","grey","green"], "sizes": ["small","medium","large","XL"],               "avg_price": 1200, "unit": "piece"},
    "water bottle": {"brands": ["Milton","Cello","Tupperware","Nalgene"],        "colors": ["blue","red","green","black","white"], "sizes": ["500ml","750ml","1L","1.5L"],                 "avg_price": 250,  "unit": "piece"},
    "tiffin box":   {"brands": ["Milton","Vaya","Cello","Tupperware"],           "colors": ["blue","red","green","grey","steel"],  "sizes": ["small","medium","large","2-tier","3-tier"],  "avg_price": 350,  "unit": "piece"},
}

PRODUCE_CATALOG: dict[str, dict] = {
    "tomato":      {"unit": "kg",     "variants": ["regular","cherry","roma"],              "avg_price": 40},
    "onion":       {"unit": "kg",     "variants": ["red","white","shallot"],                "avg_price": 30},
    "potato":      {"unit": "kg",     "variants": ["regular","baby","sweet"],               "avg_price": 25},
    "carrot":      {"unit": "kg",     "variants": None,                                     "avg_price": 50},
    "cabbage":     {"unit": "piece",  "variants": None,                                     "avg_price": 30},
    "capsicum":    {"unit": "kg",     "variants": ["green","red","yellow"],                 "avg_price": 80},
    "spinach":     {"unit": "bundle", "variants": None,                                     "avg_price": 20},
    "cucumber":    {"unit": "kg",     "variants": None,                                     "avg_price": 30},
    "brinjal":     {"unit": "kg",     "variants": None,                                     "avg_price": 40},
    "beetroot":    {"unit": "kg",     "variants": None,                                     "avg_price": 45},
    "cauliflower": {"unit": "piece",  "variants": None,                                     "avg_price": 40},
    "broccoli":    {"unit": "piece",  "variants": None,                                     "avg_price": 60},
    "garlic":      {"unit": "kg",     "variants": None,                                     "avg_price": 200},
    "ginger":      {"unit": "kg",     "variants": None,                                     "avg_price": 120},
    "lemon":       {"unit": "piece",  "variants": None,                                     "avg_price": 5},
    "apple":       {"unit": "kg",     "variants": ["red","green"],                          "avg_price": 150},
    "banana":      {"unit": "dozen",  "variants": None,                                     "avg_price": 40},
    "mango":       {"unit": "kg",     "variants": ["alphonso","kesar","totapuri"],           "avg_price": 120},
    "blueberry":   {"unit": "kg",     "variants": None,                                     "avg_price": 600},
    "strawberry":  {"unit": "kg",     "variants": None,                                     "avg_price": 200},
    "grapes":      {"unit": "kg",     "variants": ["green","black"],                        "avg_price": 80},
    "watermelon":  {"unit": "piece",  "variants": None,                                     "avg_price": 80},
    "pineapple":   {"unit": "piece",  "variants": None,                                     "avg_price": 60},
    "orange":      {"unit": "kg",     "variants": None,                                     "avg_price": 60},
    "rice":        {"unit": "kg",     "variants": ["basmati","sona masuri","ponni"],        "avg_price": 70},
    "wheat":       {"unit": "kg",     "variants": None,                                     "avg_price": 30},
    "dal":         {"unit": "kg",     "variants": ["toor","moong","masoor","chana","urad"], "avg_price": 90},
    "sugar":       {"unit": "kg",     "variants": None,                                     "avg_price": 40},
    "salt":        {"unit": "kg",     "variants": None,                                     "avg_price": 20},
    "oil":         {"unit": "litre",  "variants": ["sunflower","coconut","groundnut","mustard"],"avg_price": 130},
    "milk":        {"unit": "litre",  "variants": None,                                     "avg_price": 28},
    "egg":         {"unit": "piece",  "variants": None,                                     "avg_price": 7},
    "bread":       {"unit": "piece",  "variants": ["white","brown","multigrain"],           "avg_price": 35},
    "butter":      {"unit": "g",      "variants": None,                                     "avg_price": 0.05},
    "cheese":      {"unit": "g",      "variants": ["cheddar","mozzarella","processed"],     "avg_price": 0.04},
    "curd":        {"unit": "kg",     "variants": None,                                     "avg_price": 60},
}

# Items that exist in Agmarknet (mandi produce)
AGMARKNET_ITEMS = {
    "tomato", "onion", "potato", "carrot", "cabbage", "capsicum",
    "spinach", "cucumber", "brinjal", "beetroot", "cauliflower",
    "broccoli", "garlic", "ginger", "apple", "banana", "mango",
    "grapes", "watermelon", "orange", "rice", "wheat", "dal",
    "sugar", "lemon", "pineapple", "strawberry"
}

# ══════════════════════════════════════════════════════════════════════════════
# PRICE FETCHING — API LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _cache_get(key: str) -> float | None:
    """Return cached price if still fresh, else None."""
    if key in _price_cache:
        price, ts = _price_cache[key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return price
    return None


def _cache_set(key: str, price: float) -> None:
    _price_cache[key] = (price, time.time())


def fetch_price_agmarknet(item_name: str, state: str = "Kerala") -> float | None:
    """
    Fetch today's modal price (₹/quintal → ₹/kg) from data.gov.in Agmarknet API.
    Returns price per kg, or None on failure (silently).
    """
    cache_key = f"agmarknet:{item_name.lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    url = DATA_GOV_BASE_URL.format(resource_id=DATA_GOV_RESOURCE_ID)
    params = {
        "api-key": DATA_GOV_API_KEY,
        "format":  "json",
        "limit":   10,
        "filters[commodity]": item_name.title(),
        "filters[state]":     state,
    }
    try:
        r = requests.get(url, params=params, timeout=3)
        r.raise_for_status()
        records = r.json().get("records", [])
        if not records:
            params.pop("filters[state]", None)
            r = requests.get(url, params=params, timeout=3)
            records = r.json().get("records", [])

        prices = []
        for rec in records:
            modal = rec.get("modal_price") or rec.get("Modal_Price") or rec.get("modal price")
            if modal:
                try:
                    prices.append(float(str(modal).replace(",", "")) / 100)
                except ValueError:
                    pass

        if prices:
            avg = round(sum(prices) / len(prices), 2)
            _cache_set(cache_key, avg)
            return avg

    except Exception:
        pass  # silently fall through to next tier

    return None


def fetch_price_open_prices(item_name: str) -> float | None:
    """
    Fetch price from Open Food Facts Open Prices API (free, no key).
    Returns price in INR, or None on failure (silently).
    """
    cache_key = f"openprices:{item_name.lower()}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    url = f"{OPEN_PRICES_BASE_URL}/prices"
    params = {
        "product_name": item_name,
        "currency":     "INR",
        "page_size":    10,
    }
    try:
        r = requests.get(url, params=params, timeout=3)
        r.raise_for_status()
        items = r.json().get("items", [])
        prices = [float(i["price"]) for i in items if i.get("price")]
        if prices:
            avg = round(sum(prices) / len(prices), 2)
            _cache_set(cache_key, avg)
            return avg
    except Exception:
        pass  # silently fall through to catalog fallback

    return None


# ── Persistent learned prices file ────────────────────────────────────────────
import os
_LEARNED_PRICES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_prices.json")

def _load_learned_prices() -> dict:
    try:
        if os.path.exists(_LEARNED_PRICES_FILE):
            with open(_LEARNED_PRICES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def _save_learned_price(item_name: str, price: float) -> None:
    prices = _load_learned_prices()
    prices[item_name.lower().strip()] = price
    try:
        with open(_LEARNED_PRICES_FILE, 'w', encoding='utf-8') as f:
            json.dump(prices, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

# ── Known price estimates for common unlisted items (INR) ─────────────────────
# This table covers frequently requested items not in Agmarknet or branded catalog.
# Prices are typical Indian market averages as of 2024-25.
_FALLBACK_PRICE_TABLE: dict[str, float] = {
    "brake pad":          350.0,
    "brake pads":         350.0,
    "breakpad":           350.0,
    "mobile charger":     350.0,
    "phone charger":      350.0,
    "laptop charger":     900.0,
    "usb cable":           80.0,
    "hdmi cable":         250.0,
    "earphones":          200.0,
    "headphones":         500.0,
    "mouse":              350.0,
    "keyboard":           500.0,
    "pen drive":          350.0,
    "memory card":        400.0,
    "bulb":                60.0,
    "led bulb":            80.0,
    "tubelight":          200.0,
    "extension cord":     200.0,
    "soap":                40.0,
    "shampoo":            150.0,
    "toothpaste":          80.0,
    "toothbrush":          50.0,
    "detergent":          100.0,
    "floor cleaner":      120.0,
    "mosquito coil":       50.0,
    "napkin":             100.0,
    "tissue":              80.0,
    "matchbox":            10.0,
    "lighter":             15.0,
    "candle":              30.0,
    "umbrella":           300.0,
    "battery":             50.0,
    "screwdriver":        120.0,
    "hammer":             150.0,
    "nail":                20.0,
    "tape":                30.0,
    "rope":                60.0,
    "bucket":             120.0,
    "mug":                 40.0,
    "broom":               80.0,
    "mop":                150.0,
}


def fetch_price_llm_estimate(item_name: str) -> float | None:
    """
    Tier 4: Price estimate for unknown items.
    Step A — check learned prices saved from previous sessions.
    Step B — check built-in fallback table.
    Step C — ask Groq LLM and save the result permanently.
    """
    key = item_name.lower().strip()

    # Step A: previously learned prices (saved to disk — instant)
    learned = _load_learned_prices()
    if key in learned:
        return learned[key]
    for lkey, lprice in learned.items():
        if lkey in key or key in lkey:
            return lprice

    # Step B: built-in fallback table
    if key in _FALLBACK_PRICE_TABLE:
        price = _FALLBACK_PRICE_TABLE[key]
        _save_learned_price(key, price)   # promote to learned
        return price
    for table_key, table_price in _FALLBACK_PRICE_TABLE.items():
        if table_key in key or key in table_key:
            _save_learned_price(key, table_price)
            return table_price

    # Step C: ask Groq LLM — same client & model already working
    cache_key = f"llm_estimate:{key}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    prompt = (
        f"Indian retail price for \"{item_name}\" in INR. "
        f"Single integer only. No words. No symbol. Example: 350"
    )
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You output only a single integer number. Nothing else."},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = response.choices[0].message.content.strip()
        match = re.search(r"\d+", raw)
        if match:
            price = float(match.group())
            if 1 <= price <= 500000:
                _cache_set(cache_key, price)
                _save_learned_price(key, price)   # ← save permanently to disk
                return price
    except Exception:
        pass

    return None


def get_live_price(item_name: str) -> tuple[float | None, str]:
    """
    Master price resolver with 4-tier fallback:
      1. data.gov.in Agmarknet  (produce / mandi items)
      2. Open Prices API         (packaged / branded items)
      3. Static catalog
      4. LLM price estimate      (unknown items — always returns a price)

    Returns (price, source_label).
    """
    key = item_name.lower().strip()

    # Tier 1 — Agmarknet (best for Indian produce)
    if key in AGMARKNET_ITEMS:
        price = fetch_price_agmarknet(key)
        if price:
            return price, "🌐 live (Agmarknet)"

    # Tier 2 — Open Prices API (branded/packaged)
    price = fetch_price_open_prices(key)
    if price:
        return price, "🌐 live (Open Prices)"

    # Tier 3 — Static catalog
    if key in BRANDED_CATALOG:
        return BRANDED_CATALOG[key]["avg_price"], "📦 catalog"
    if key in PRODUCE_CATALOG:
        return PRODUCE_CATALOG[key]["avg_price"], "📦 catalog"

    # Tier 4 — LLM estimate (catches everything else — brake pads, chargers, etc.)
    price = fetch_price_llm_estimate(item_name)
    if price:
        return price, "🤖"

    return None, "❓ unknown"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  (unchanged from v4, price source added)
# ══════════════════════════════════════════════════════════════════════════════

def get_avg_price(item_name: str) -> float | None:
    price, _ = get_live_price(item_name)
    return price


def calc_item_total(item: dict) -> float:
    avg = item.get("avg_price_per_unit") or get_avg_price(item.get("item_name", ""))
    if avg is None:
        return 0.0
    return avg * float(item.get("quantity", 1))


def build_catalog_summary() -> str:
    sections = []
    branded_lines = ["BRANDED PRODUCTS CATALOG (ask for brand + color when missing):"]
    for name, d in BRANDED_CATALOG.items():
        parts = [f"  * {name.title()}  [avg price: ₹{d['avg_price']} per {d['unit']}]"]
        parts.append(f"    Brands: {', '.join(d['brands'])}")
        parts.append(f"    Colors: {', '.join(d['colors'])}")
        if d["sizes"]:
            parts.append(f"    Sizes : {', '.join(d['sizes'])}")
        branded_lines.append("\n".join(parts))
    sections.append("\n\n".join(branded_lines))

    produce_lines = ["\nPRODUCE / GROCERY CATALOG (NO brand/color — only quantity, unit, optional variant):"]
    for name, d in PRODUCE_CATALOG.items():
        line = f"  * {name.title()}  [unit: {d['unit']}, avg price: ₹{d['avg_price']} per {d['unit']}]"
        if d["variants"]:
            line += f"  variants: {', '.join(d['variants'])}"
        produce_lines.append(line)
    sections.append("\n".join(produce_lines))
    return "\n\n".join(sections)


CATALOG_SUMMARY = build_catalog_summary()

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""
You are a universal Shopping Memory Assistant that helps users manage ANY kind of
shopping list — groceries, fruits, vegetables, stationery, electronics, household
items, or anything else. You NEVER refuse to add an item.

You MUST always respond with a single valid JSON object and nothing else.
No markdown, no code fences, no extra text — pure JSON only.

TYPO HANDLING: Silently correct spelling mistakes and proceed.
Examples: onian=onion, bluebarry=blueberry, breakpad=brake pad, grosorys=groceries.

{CATALOG_SUMMARY}

════════════════════════════════════════
ITEM TYPE RULES
════════════════════════════════════════

A) PRODUCE / FOOD (tomato, onion, rice, milk, egg, etc.):
   - No brand, no color. Only: item_name, quantity, unit, variant (optional).
   - Set brand=null, color=null, missing_fields=[].
   - Add immediately, no questions about brand/color.
   - avg_price_per_unit will be fetched live from API.

B) BRANDED PRODUCTS (pen, notebook, bag, calculator, etc.):
   - ALWAYS add immediately even if brand/color missing.
   - After adding, ask for missing details in chat_response.
   - avg_price_per_unit will be fetched live from API.

C) UNKNOWN ITEMS not in catalog (brake pads, laptop charger, etc.):
   - If food/consumable: treat as Type A.
   - If manufactured product: treat as Type B, add immediately, ask brand/color.
   - Set avg_price_per_unit = null for unknown items.

════════════════════════════════════════
INTENTS
════════════════════════════════════════

INTENT: add_item
{{
  "intent": "add_item",
  "chat_response": "...",
  "item": {{
    "item_name": "string",
    "brand": "string or null",
    "color": "string or null",
    "variant": "string or null",
    "quantity": 1,
    "unit": "kg/litre/piece/g/dozen/bundle/null",
    "avg_price_per_unit": null
  }},
  "missing_fields": []
}}

INTENT: add_items_bulk
{{
  "intent": "add_items_bulk",
  "chat_response": "Summary of what was added and what still needs info",
  "items": [
    {{"item_name":"tomato","brand":null,"color":null,"variant":null,"quantity":1,"unit":"kg","avg_price_per_unit":null,"missing_fields":[]}},
    {{"item_name":"pen","brand":null,"color":null,"variant":null,"quantity":1,"unit":"piece","avg_price_per_unit":null,"missing_fields":["brand","color"]}}
  ]
}}

INTENT: update_quantity
{{"intent":"update_quantity","chat_response":"...","item_name":"onion","new_quantity":2}}

INTENT: remove_item
{{
  "intent": "remove_item",
  "chat_response": "...",
  "filters": {{"item_name":"...","brand":null,"color":null}},
  "ambiguity": false
}}

INTENT: show_list
Use when user asks to see list, total, bill, amount, price, cost, how much, grand total.
{{"intent":"show_list","chat_response":"..."}}

INTENT: confirm_order
{{"intent":"confirm_order","chat_response":"Order confirmed! Here is your summary."}}

INTENT: recommend
{{
  "intent": "recommend",
  "chat_response": "Thoughtful recommendation referencing catalog options",
  "criteria": {{"item_name":null,"use_case":null,"budget":null}}
}}

INTENT: chat
{{"intent":"chat","chat_response":"...","item":null,"filters":null}}

GLOBAL RULES:
- ONLY output valid JSON. No extra text.
- null for unknown fields, never "unknown" or "".
- quantity defaults to 1.
- missing_fields = [] for all food/produce items.
- Leave avg_price_per_unit as null — the app will fetch it live from APIs.
"""

# ── In-memory state ────────────────────────────────────────────────────────────
shopping_list:        list[dict] = []
conversation_history: list[dict] = []


# ── LLM call ──────────────────────────────────────────────────────────────────
def call_llm(user_message: str) -> dict:
    conversation_history.append({"role": "user", "content": user_message})
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history,
        temperature=0.2,
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"intent": "chat", "chat_response": raw, "item": None, "filters": None}
    conversation_history.append({"role": "assistant", "content": json.dumps(parsed)})
    return parsed


# ── Catalog hint ───────────────────────────────────────────────────────────────
def catalog_hint(item_name: str | None, missing: list[str]) -> str:
    if not item_name or not missing:
        return ""
    entry = BRANDED_CATALOG.get(item_name.lower())
    if not entry:
        return ""
    parts = []
    if "brand" in missing:
        parts.append(f"   🏷  Brands  : {', '.join(entry['brands'])}")
    if "color" in missing:
        parts.append(f"   🎨  Colors  : {', '.join(entry['colors'])}")
    if "size" in missing and entry.get("sizes"):
        parts.append(f"   📐  Sizes   : {', '.join(entry['sizes'])}")
    return "\n".join(parts)


def item_matches(item: dict, filters: dict) -> bool:
    for key, value in filters.items():
        if value is None:
            continue
        if item.get(key, "").lower() != value.lower():
            return False
    return True


def fmt_price(amount: float | None) -> str:
    if amount is None:
        return "—"
    return f"₹{amount:,.2f}"


def _add_one_item(item: dict, missing: list[str]) -> None:
    """Resolve live price then add item to list.
    Always calls get_live_price regardless of what the LLM sent,
    so unknown items always reach Tier 4 (AI estimate).
    """
    price, source = get_live_price(item.get("item_name", ""))
    item["avg_price_per_unit"] = price
    item["_price_source"] = source

    shopping_list.append(item)
    # No row printed here — assistant's chat_response already says what was added.
    # Full details shown only on show_list / confirm_order.


# ── Display helpers ────────────────────────────────────────────────────────────
def print_shopping_list(title: str = "Shopping List") -> float:
    grand_total = 0.0
    if not shopping_list:
        print("   📋  Your shopping list is empty.")
        return 0.0

    print(f"\n   📋  {title}")
    print(f"   {'#':<4} {'Item':<16} {'Qty':<8} {'Unit':<7} {'Brand':<14} {'Color':<12} {'Avg/unit':<12} {'Subtotal':<12} {'Source'}")
    print("   " + "─" * 100)

    for idx, item in enumerate(shopping_list, 1):
        name     = item.get("item_name", "—").title()
        qty      = item.get("quantity", 1)
        unit     = item.get("unit") or "—"
        brand    = item.get("brand") or "—"
        color    = item.get("color") or "—"
        price_u  = item.get("avg_price_per_unit")
        subtotal = calc_item_total(item)
        source   = item.get("_price_source", "")
        grand_total += subtotal

        pu_str  = fmt_price(price_u) if price_u else "—"
        sub_str = fmt_price(subtotal) if price_u else "—"

        print(f"   {idx:<4} {name:<16} {str(qty):<8} {unit:<7} {brand:<14} {color:<12} {pu_str:<12} {sub_str:<12} {source}")

    print("   " + "─" * 100)
    unknown_count = sum(1 for i in shopping_list if not i.get("avg_price_per_unit"))
    print(f"   {'':>78} TOTAL: {fmt_price(grand_total)}")
    if unknown_count:
        print(f"   ⚠  {unknown_count} item(s) have unknown prices — actual total may be higher.")
    return grand_total


# ── Intent handlers ────────────────────────────────────────────────────────────
def handle_add_item(data: dict) -> None:
    item    = data.get("item") or {}
    missing = data.get("missing_fields") or []
    print(f"\n🛒  Assistant: {data['chat_response']}")
    _add_one_item(item, missing)


def handle_add_items_bulk(data: dict) -> None:
    items = data.get("items") or []
    print(f"\n🛒  Assistant: {data['chat_response']}")
    for itm in items:
        missing = itm.pop("missing_fields", [])
        _add_one_item(itm, missing)


def handle_update_quantity(data: dict) -> None:
    item_name    = (data.get("item_name") or "").lower().strip()
    new_quantity = data.get("new_quantity")
    print(f"\n🛒  Assistant: {data['chat_response']}")
    if not item_name or new_quantity is None:
        print("   ⚠  Could not determine which item or new quantity.")
        return
    matched = [i for i in shopping_list if item_name in i.get("item_name", "").lower()]
    if not matched:
        print(f"   ℹ  '{item_name}' not found in your list.")
        return
    for i in matched:
        i["quantity"] = new_quantity
    print(f"   ✅  Updated '{item_name}' quantity to {new_quantity}.")


def handle_remove_item(data: dict) -> None:
    filters   = data.get("filters") or {}
    ambiguous = data.get("ambiguity", False)
    matches   = [i for i in shopping_list if item_matches(i, filters)]
    print(f"\n🛒  Assistant: {data['chat_response']}")
    if not matches:
        print("   ℹ  No matching items found.")
        return
    if ambiguous and len(matches) > 1:
        print("   ⚠  Multiple matches — be more specific:")
        for idx, m in enumerate(matches, 1):
            print(f"      {idx}. {m}")
        return
    removed = [i.get("item_name") for i in matches]
    for m in matches:
        shopping_list.remove(m)
    print(f"   🗑  Removed: {removed}")


def handle_show_list(data: dict) -> None:
    total = sum(calc_item_total(i) for i in shopping_list)
    count = len(shopping_list)
    if shopping_list:
        print(f"\n🛒  Assistant: Here are your {count} item(s). Estimated total: {fmt_price(total)}")
    else:
        print(f"\n🛒  Assistant: Your shopping list is empty.")
    print_shopping_list()


def handle_confirm_order(data: dict) -> None:
    print(f"\n🛒  Assistant: {data['chat_response']}")
    if not shopping_list:
        print("   ℹ  Your shopping list is empty — nothing to confirm.")
        return

    total = print_shopping_list("Order Confirmation")

    print(f"\n   {'='*100}")
    print(f"   🎯  ORDER CONFIRMED")
    print(f"   📦  Total items     : {len(shopping_list)}")
    print(f"   💰  Estimated total : {fmt_price(total)}")
    live_count     = sum(1 for i in shopping_list if "live"     in i.get("_price_source",""))
    catalog_count  = sum(1 for i in shopping_list if "catalog"  in i.get("_price_source",""))
    ai_count       = sum(1 for i in shopping_list if "AI"       in i.get("_price_source",""))
    print(f"   🌐  Live prices     : {live_count} item(s)")
    print(f"   📦  Catalog prices  : {catalog_count} item(s)")
    if ai_count:
        print(f"   🤖  AI estimates    : {ai_count} item(s)  ← approximate, verify before buying")
    print(f"   {'='*100}")
    print("   ✅  Your order has been noted. Happy shopping!\n")

    shopping_list.clear()
    conversation_history.append({
        "role": "assistant",
        "content": json.dumps({"note": "Order confirmed and list cleared."})
    })


def handle_recommend(data: dict) -> None:
    print(f"\n🛒  Assistant: {data['chat_response']}")
    active = {k: v for k, v in (data.get("criteria") or {}).items() if v}
    if active:
        print(f"   💡  Criteria: {active}")


# Keywords that mean the user wants the total price
_TOTAL_KEYWORDS = {"total", "totel", "totol", "tota", "bill", "amount", "sum",
                   "how much", "price", "cost", "grand total", "final"}

def handle_chat(data: dict) -> None:
    # If LLM routed to chat but user was asking for total, override with real total
    response_text = data.get("chat_response", "")
    if shopping_list and any(k in response_text.lower() for k in _TOTAL_KEYWORDS):
        total = sum(calc_item_total(i) for i in shopping_list)
        print(f"\n🛒  Assistant: Your current estimated total is {fmt_price(total)} for {len(shopping_list)} item(s).")
        print_shopping_list()
    else:
        print(f"\n🛒  Assistant: {response_text}")


HANDLERS = {
    "add_item":        handle_add_item,
    "add_items_bulk":  handle_add_items_bulk,
    "update_quantity": handle_update_quantity,
    "remove_item":     handle_remove_item,
    "show_list":       handle_show_list,
    "confirm_order":   handle_confirm_order,
    "recommend":       handle_recommend,
    "chat":            handle_chat,
}


# ── Main loop ──────────────────────────────────────────────────────────────────
def main() -> None:
    branded = ", ".join(k.title() for k in BRANDED_CATALOG)
    produce = ", ".join(k.title() for k in list(PRODUCE_CATALOG)[:10]) + "..."
    print("=" * 70)
    print("  🛍   Shopping Memory Assistant  v5  (powered by Groq + Live APIs)")
    print("=" * 70)
    print(f"  Branded  : {branded}")
    print(f"  Groceries: {produce}")
    print()
    print("  💰  Price sources (in priority order):")
    print("      1. 🌐 data.gov.in Agmarknet  — live mandi prices (produce/grains)")
    print("      2. 🌐 Open Prices API         — live packaged item prices")
    print("      3. 📦 Static catalog          — built-in average prices")
    print("      4. 🤖 AI estimate             — Groq LLM price for unknown items")
    print()
    print("  ℹ  To get full live Agmarknet data, replace DATA_GOV_API_KEY")
    print("     with your free key from https://data.gov.in")
    print()
    print("  Say 'confirm order' to finalize. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye! Happy shopping! 🛍")
            break
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "bye"}:
            print("\n🛒  Assistant: Goodbye! Happy shopping! 🛍")
            break
        try:
            data   = call_llm(user_input)
            intent = data.get("intent", "chat")
            HANDLERS.get(intent, handle_chat)(data)
        except Exception as exc:
            print(f"\n⚠  Error: {exc}")
        print()


if __name__ == "__main__":
    main()
