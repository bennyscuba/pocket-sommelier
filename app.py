import streamlit as st
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from supabase import create_client, Client
from PIL import Image

# Initialize API clients
genai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])

# Define the structured data the AI must return
class WineData(BaseModel):
    wine_type: str = Field(description="Red, White, Rose, or Sparkling")
    producer: str = Field(description="The winery or estate")
    vintage: str = Field(description="The year, or 'NV' for non-vintage")
    varietal: str = Field(description="The grape or blend type")
    region: str = Field(description="The geographic region")
    flavor_notes: str = Field(description="3-4 distinct flavor notes")
    tannin_level: str = Field(description="Low, Medium, or High")
    acidity: str = Field(description="Low, Medium, or High")
    body_type: str = Field(description="Light, Medium, or Full")
    drink_window: str = Field(description="e.g., 'Drink Now' or 'Hold until 2028'")

st.title("🍷 Pocket Sommelier")

tab1, tab2, tab3 = st.tabs(["Scan & Add", "Sommelier Pairing", "Manage Cellar"])

# --- TAB 1: SCAN & ADD ---
with tab1:
    st.write("Scan a label to automatically extract and infer its profile.")
    picture = st.camera_input("Scan Wine Label")
    
if picture:
        with st.spinner("Analyzing label and inferring structural profile..."):
            img = Image.open(picture)
            
            # Convert PIL image to JPEG bytes for google-genai SDK
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            img_bytes = img_byte_arr.getvalue()

            prompt = "Extract the wine type (Red, White, Rose, or Sparkling), producer, vintage, varietal, and region. Then, infer the flavor notes, tannin level, acidity, body type, and ideal drink window based on the wine's profile."
            
            response = genai_client.models.generate_content(
                model='gemini-flash-latest',
                contents=[
                    types.Part.from_bytes(
                        data=img_bytes,
                        mime_type='image/jpeg',
                    ),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=WineData,
                ),
            )
            
            data = json.loads(response.text)
            st.success(f"Identified: {data['vintage']} {data['producer']} {data['varietal']}")
            
            # Review and add manual logistics (allow blanks)
            with st.form("add_wine_form"):
                st.subheader("Optional Logistics")
                col1, col2 = st.columns(2)
                with col1:
                    rack_loc = st.text_input("Rack Location", placeholder="e.g., Top row")
                    price = st.number_input("Purchase Price ($)", min_value=0.0, step=1.0, value=None)
                with col2:
                    source = st.text_input("Source", placeholder="e.g., Rocks and Acid")
                
                submitted = st.form_submit_button("Save to Cellar")
                if submitted:
                    supabase.table("wines").insert({
                        "wine_type": data["wine_type"],
                        "producer": data["producer"],
                        "vintage": data["vintage"],
                        "varietal": data["varietal"],
                        "region": data["region"],
                        "flavor_notes": data["flavor_notes"],
                        "tannin_level": data["tannin_level"],
                        "acidity": data["acidity"],
                        "body_type": data["body_type"],
                        "drink_window": data["drink_window"],
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
        # Fetch active inventory
        response = supabase.table("wines").select("*").eq("is_consumed", False).gt("quantity", 0).execute()
        inventory = response.data
        
        if not inventory:
            st.warning("Your rack is currently empty.")
        else:
            with st.spinner("Finding the perfect structural match..."):
                # Pass structural metrics to the AI
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
    # Fetch active inventory to manage
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
