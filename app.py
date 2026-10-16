import streamlit as st
import json
from google import genai
from google.genai import types
from supabase import create_client, Client
from PIL import Image

# Initialize API clients
genai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

st.title("🍷 Pocket Sommelier")

tab1, tab2, tab3 = st.tabs(["Scan & Add", "Sommelier Pairing", "Manage Cellar"])

# --- TAB 1: SCAN & ADD ---
with tab1:
    st.write("Scan a label to automatically extract and infer its profile.")
    picture = st.camera_input("Scan Wine Label")
    
    if picture:
        with st.spinner("Analyzing label and inferring structural profile..."):
            img = Image.open(picture)
            
            # Drop the alpha channel to prevent malformed JPEGs
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
                
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()

            prompt = """Analyze this wine label image. 
            Return ONLY a valid JSON object. Do not use markdown blocks, backticks, or conversational text. Use these exact keys:
            {
                "wine_type": "Red, White, Rose, or Sparkling",
                "producer": "winery name",
                "vintage": "year or NV",
                "varietal": "grape or blend",
                "region": "region",
                "flavor_notes": "3-4 flavor notes",
                "tannin_level": "Low, Medium, or High",
                "acidity": "Low, Medium, or High",
                "body_type": "Light, Medium, or Full",
                "drink_window": "e.g. Drink Now"
            }"""
            
            response = genai_client.models.generate_content(
                model='gemini-flash-latest',
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt,
                ],
            )
            
            raw_text = response.text.strip()
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.startswith("```"):
                raw_text = raw_text[3:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
                
            data = json.loads(raw_text.strip())
            st.success(f"Identified: {data.get('vintage')} {data.get('producer')} {data.get('varietal')}")
            
            # Review and add manual logistics
            with st.form("add_wine_form"):
                st.subheader("Optional Logistics")
                col1, col2 = st.columns(2)
                with col1:
                    rack_loc = st.text_input("Rack Location", placeholder="e.g., Top row")
                    price = st.number_input("Purchase Price ($)", min_value=0.0, step=1.0, value=None)
                with col2:
                    source = st.text_input("Source", placeholder="e.g., Vin Rouge, Rue Cler")
                
                submitted = st.form_submit_button("Save to Cellar")
                if submitted:
                    supabase.table("wines").insert({
                        "wine_type": data.get("wine_type", "Unknown"),
                        "producer": data.get("producer", "Unknown"),
                        "vintage": data.get("vintage", "Unknown"),
                        "varietal": data.get("varietal", "Unknown"),
                        "region": data.get("region", "Unknown"),
                        "flavor_notes": data.get("flavor_notes", "Unknown"),
                        "tannin_level": data.get("tannin_level", "Unknown"),
                        "acidity": data.get("acidity", "Unknown"),
                        "body_type": data.get("body_type", "Unknown"),
                        "drink_window": data.get("drink_window", "Unknown"),
                        "rack_location": rack_loc,
                        "source": source,
                        "purchase_price": price,
                        "quantity": 1
                    }).execute()
                    st.success("Bottle added to inventory!")

# --- TAB 2: SOMMELIER PAIRING ---
with tab2:
    meal = st.text_input("What's on the menu?", placeholder="e.g., Smoked brisket, fresh fish ceviche, mushroom risotto")
    if st.button("Find Pairing") and meal:
        response = supabase.table("wines").select("*").eq("is_consumed", False).gt("quantity", 0).execute()
        inventory = response.data
        
        if not inventory:
            st.warning("Your rack is currently empty.")
        else:
            with st.spinner("Finding the perfect structural match..."):
                inventory_str = "\n".join([f"- {w['vintage']} {w['producer']} {w['varietal']} | Tannins: {w['tannin_level']}, Acidity: {w['acidity']}, Body: {w['body_type']}, Notes: {w['flavor_notes']}" for w in inventory])
                
                pairing_prompt = f"I am cooking: {meal}. Here is my active wine inventory with their structural metrics:\n{inventory_str}\n\nRecommend the single best match. Explain why the acidity, tannins, and body pair well with the specific preparation of this dish."
                
                pairing_response = genai_client.models.generate_content(
                    model='gemini-flash-latest',
                    contents=pairing_prompt,
                )
                st.write(pairing_response.text)

# --- TAB 3: MANAGE CELLAR ---
with tab3:
    st.write("Log notes and update quantities after a meal.")
    response = supabase.table("wines").select("*").eq("is_consumed", False).gt("quantity", 0).execute()
    wines = response.data
    
    if not wines:
        st.info("No active bottles to manage.")
    else:
        for w in wines:
            with st.expander(f"{w['vintage']} {w['producer']} {w['varietal']}"):
                st.write(f"**Location:** {w['rack_location'] or 'Unassigned'} | **Window:** {w['drink_window']}")
                
                notes = st.text_area("Tasting Notes", key=f"notes_{w['wine_id']}", placeholder="How did it pair?")
                if st.button("Mark Consumed", key=f"btn_{w['wine_id']}"):
                    supabase.table("wines").update({
                        "quantity": w['quantity'] - 1,
                        "is_consumed": True if (w['quantity'] - 1) == 0 else False,
                        "post_drink_notes": notes
                    }).eq("wine_id", w['wine_id']).execute()
                    st.success("Cellar updated! Refreshing...")
                    st.rerun()
