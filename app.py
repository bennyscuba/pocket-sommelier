import io

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from supabase import Client, create_client
from PIL import Image


# -----------------------------------------------------------------------------
# API clients
# -----------------------------------------------------------------------------
genai_client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"],
)


# -----------------------------------------------------------------------------
# Structured Gemini responses
# -----------------------------------------------------------------------------
class WineData(BaseModel):
    wine_type: str = Field(description="Red, White, Rose, or Sparkling")
    producer: str = Field(description="The winery or estate shown on the label")
    vintage: str = Field(description="The four-digit year, or NV for non-vintage")
    varietal: str = Field(description="The grape variety or blend")
    region: str = Field(description="The geographic wine region")
    flavor_notes: str = Field(description="Three or four likely flavor notes")
    tannin_level: str = Field(description="Low, Medium, or High")
    acidity: str = Field(description="Low, Medium, or High")
    body_type: str = Field(description="Light, Medium, or Full")
    drink_window: str = Field(description="For example, Drink Now or Hold until 2028")
    quality_score: int = Field(
        ge=1,
        le=5,
        description=(
            "Subjective cellar quality from 1 to 5. Use 1 for poor, 2 for below "
            "average, 3 for good everyday quality, 4 for excellent, and 5 for "
            "exceptional or special-occasion quality."
        ),
    )
    decant_minutes: int = Field(
        ge=0,
        le=240,
        description=(
            "Recommended decanting time in whole minutes. Use 0 when no decanting "
            "is recommended."
        ),
    )


class PairingRecommendation(BaseModel):
    wine_id: int = Field(description="The wine_id of the selected wine")
    reason: str = Field(description="Why this specific wine pairs with the meal")
    serving_temperature: str = Field(description="Recommended serving temperature")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def analyze_wine_label(picture) -> dict:
    picture.seek(0)
    img = Image.open(picture).convert("RGB")

    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="JPEG", quality=90, optimize=True)

    prompt = """
    Examine this wine label.

    Extract the wine type, producer, vintage, varietal, and region. Then infer
    likely flavor notes, tannin level, acidity, body type, ideal drink window,
    a subjective quality score from 1 to 5, and recommended decanting time.

    Quality scale:
    1 = poor; generally not worth drinking
    2 = below average; rarely selected
    3 = good; appropriate for regular drinking
    4 = excellent; a higher-quality bottle
    5 = exceptional; special-occasion quality

    For decant_minutes, return a whole number from 0 to 240. Use 0 if the wine
    should not be decanted. Consider wine style, grape, region, and vintage.

    Use NV if no vintage is visible. Do not invent label details that cannot be
    read. Structural characteristics, quality, drink window, and decanting are
    informed estimates rather than facts printed on the label.
    """

    response = genai_client.models.generate_content(
        model="gemini-flash-latest",
        contents=[
            types.Part.from_bytes(
                data=img_byte_arr.getvalue(),
                mime_type="image/jpeg",
            ),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=WineData,
            temperature=0.1,
        ),
    )

    return WineData.model_validate_json(response.text).model_dump()


def quality_label(score: int) -> str:
    labels = {
        1: "1 - Poor",
        2: "2 - Below average",
        3: "3 - Good / regular drinking",
        4: "4 - Excellent",
        5: "5 - Exceptional / special occasion",
    }
    return labels.get(score, str(score))


# -----------------------------------------------------------------------------
# App shell
# -----------------------------------------------------------------------------
st.title("🍷 Pocket Sommelier")

tab1, tab2, tab3 = st.tabs(
    ["Scan & Add", "Sommelier Pairing", "Manage Cellar"]
)


# -----------------------------------------------------------------------------
# TAB 1: SCAN & ADD
# -----------------------------------------------------------------------------
with tab1:
    st.subheader("Scan and Add a Bottle")
    st.write("Scan a label, review the result, and save the bottle to your cellar.")

    picture = st.camera_input("Scan Wine Label")

    if picture is not None and st.button("Analyze Label", type="primary"):
        try:
            with st.spinner("Analyzing label and inferring its profile..."):
                st.session_state["wine_data"] = analyze_wine_label(picture)
        except ValidationError as exc:
            st.error("Gemini returned data that did not match the wine schema.")
            st.code(str(exc))
        except Exception as exc:
            st.error("The wine label could not be analyzed.")
            st.code(f"{type(exc).__name__}: {exc}")

    data = st.session_state.get("wine_data")

    if data:
        st.success(
            f"Identified: {data['vintage']} {data['producer']} {data['varietal']}"
        )

        with st.form("add_wine_form", clear_on_submit=True):
            st.subheader("Review Wine Details")

            col1, col2 = st.columns(2)
            with col1:
                producer = st.text_input("Producer", value=data["producer"])
                vintage = st.text_input("Vintage", value=data["vintage"])
                varietal = st.text_input("Varietal", value=data["varietal"])
                wine_type = st.selectbox(
                    "Wine Type",
                    ["Red", "White", "Rose", "Sparkling"],
                    index=["Red", "White", "Rose", "Sparkling"].index(data["wine_type"])
                    if data["wine_type"] in ["Red", "White", "Rose", "Sparkling"]
                    else 0,
                )
                region = st.text_input("Region", value=data["region"])

            with col2:
                flavor_notes = st.text_area(
                    "Flavor Notes", value=data["flavor_notes"]
                )
                tannin_level = st.selectbox(
                    "Tannin Level",
                    ["Low", "Medium", "High"],
                    index=["Low", "Medium", "High"].index(data["tannin_level"])
                    if data["tannin_level"] in ["Low", "Medium", "High"]
                    else 1,
                )
                acidity = st.selectbox(
                    "Acidity",
                    ["Low", "Medium", "High"],
                    index=["Low", "Medium", "High"].index(data["acidity"])
                    if data["acidity"] in ["Low", "Medium", "High"]
                    else 1,
                )
                body_type = st.selectbox(
                    "Body",
                    ["Light", "Medium", "Full"],
                    index=["Light", "Medium", "Full"].index(data["body_type"])
                    if data["body_type"] in ["Light", "Medium", "Full"]
                    else 1,
                )
                drink_window = st.text_input(
                    "Drink Window", value=data["drink_window"]
                )

            st.subheader("Cellar Classification")
            class_col1, class_col2 = st.columns(2)
            with class_col1:
                quality_score = st.selectbox(
                    "Subjective Quality",
                    options=[1, 2, 3, 4, 5],
                    index=max(0, min(4, int(data["quality_score"]) - 1)),
                    format_func=quality_label,
                    help="Gemini suggests a score, but you make the final classification.",
                )
            with class_col2:
                decant_minutes = st.number_input(
                    "Recommended Decant Time (minutes)",
                    min_value=0,
                    max_value=240,
                    step=5,
                    value=int(data["decant_minutes"]),
                    help="Use 0 when no decanting is recommended.",
                )

            st.subheader("Optional Logistics")
            logistics_col1, logistics_col2 = st.columns(2)
            with logistics_col1:
                rack_loc = st.text_input(
                    "Rack Location", placeholder="e.g., Top row"
                )
                price = st.number_input(
                    "Purchase Price ($)", min_value=0.0, step=1.0, value=None
                )
            with logistics_col2:
                source = st.text_input(
                    "Source", placeholder="e.g., Vin Rouge, Rue Cler"
                )
                quantity = st.number_input(
                    "Quantity", min_value=1, step=1, value=1
                )

            submitted = st.form_submit_button("Save to Cellar", type="primary")

            if submitted:
                record = {
                    "wine_type": wine_type,
                    "producer": producer,
                    "vintage": vintage,
                    "varietal": varietal,
                    "region": region,
                    "flavor_notes": flavor_notes,
                    "tannin_level": tannin_level,
                    "acidity": acidity,
                    "body_type": body_type,
                    "drink_window": drink_window,
                    "quality_score": int(quality_score),
                    "decant_minutes": int(decant_minutes),
                    "rack_location": rack_loc or None,
                    "source": source or None,
                    "purchase_price": price,
                    "quantity": int(quantity),
                    "is_consumed": False,
                }

                try:
                    supabase.table("wines").insert(record).execute()
                    st.session_state.pop("wine_data", None)
                    st.success("Bottle added to inventory!")
                except Exception as exc:
                    st.error("The bottle could not be saved.")
                    st.code(f"{type(exc).__name__}: {exc}")


# -----------------------------------------------------------------------------
# TAB 2: SOMMELIER PAIRING
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🍽️ Find a Wine for Dinner")
    st.write(
        "Describe the meal and choose the quality level you want to drink. "
        "Purchase price is not provided to Gemini and does not affect its choice."
    )

    meal = st.text_area(
        "What's on the menu?",
        placeholder="e.g., Grilled ribeye with mushrooms and roasted potatoes",
        key="meal_description",
    )

    quality_group = st.selectbox(
        "Which quality level should I search?",
        [
            "3 - Good / regular drinking",
            "4 - Excellent",
            "5 - Exceptional / special occasion",
            "1-2 - Lower-rated bottles",
        ],
        help="Only bottles in the selected group are sent to Gemini.",
    )

    quality_map = {
        "1-2 - Lower-rated bottles": [1, 2],
        "3 - Good / regular drinking": [3],
        "4 - Excellent": [4],
        "5 - Exceptional / special occasion": [5],
    }

    if st.button("Find Best Pairing", type="primary"):
        if not meal.strip():
            st.warning("Tell me what you're eating first.")
        else:
            try:
                selected_scores = quality_map[quality_group]
                query = (
                    supabase.table("wines")
                    .select("*")
                    .eq("is_consumed", False)
                    .gt("quantity", 0)
                    .in_("quality_score", selected_scores)
                )
                response = query.execute()
                inventory = response.data

                if not inventory:
                    st.warning(
                        f"No available bottles were found in quality group "
                        f"{quality_group}. Try another level."
                    )
                    st.session_state.pop("pairing_result", None)
                else:
                    inventory_str = "\n\n".join(
                        [
                            (
                                f"WINE ID: {w['wine_id']}\n"
                                f"Wine: {w['vintage']} {w['producer']} {w['varietal']}\n"
                                f"Type: {w['wine_type']}\n"
                                f"Region: {w['region']}\n"
                                f"Quality score: {w['quality_score']}\n"
                                f"Tannins: {w['tannin_level']}\n"
                                f"Acidity: {w['acidity']}\n"
                                f"Body: {w['body_type']}\n"
                                f"Flavor notes: {w['flavor_notes']}\n"
                                f"Drink window: {w['drink_window']}\n"
                                f"Stored decant recommendation: {w.get('decant_minutes', 0)} minutes\n"
                                f"Quantity: {w['quantity']}"
                            )
                            for w in inventory
                        ]
                    )

                    pairing_prompt = f"""
                    You are my personal sommelier.

                    DINNER:
                    {meal}

                    AVAILABLE WINES IN THE USER'S SELECTED QUALITY GROUP:
                    {inventory_str}

                    Choose the single best structural pairing from this inventory.
                    You must choose one exact WINE ID listed above. Do not recommend
                    a wine outside this inventory. Ignore prestige and monetary value;
                    purchase prices are intentionally unavailable.

                    Consider acidity versus richness, tannin versus protein and cooking
                    method, wine body versus dish weight, flavor compatibility, sauce,
                    preparation, and drink window. Explain why the exact selected wine
                    is the strongest match and provide a serving temperature.
                    """

                    with st.spinner("Searching your cellar for the best match..."):
                        pairing_response = genai_client.models.generate_content(
                            model="gemini-flash-latest",
                            contents=pairing_prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=PairingRecommendation,
                                temperature=0.1,
                            ),
                        )

                    pairing = PairingRecommendation.model_validate_json(
                        pairing_response.text
                    )
                    selected_wine = next(
                        (
                            w
                            for w in inventory
                            if int(w["wine_id"]) == int(pairing.wine_id)
                        ),
                        None,
                    )

                    if selected_wine is None:
                        st.error(
                            "The sommelier selected a wine that was not in the "
                            "filtered cellar results. Please try again."
                        )
                    else:
                        st.session_state["pairing_result"] = {
                            "wine": selected_wine,
                            "reason": pairing.reason,
                            "temperature": pairing.serving_temperature,
                            "meal": meal,
                            "quality_group": quality_group,
                        }

            except ValidationError as exc:
                st.error("Gemini returned a pairing that did not match the schema.")
                st.code(str(exc))
            except Exception as exc:
                st.error("I couldn't search the cellar.")
                st.code(f"{type(exc).__name__}: {exc}")

    if "pairing_result" in st.session_state:
        result = st.session_state["pairing_result"]
        wine = result["wine"]

        st.markdown("---")
        st.subheader("🍷 My Recommendation")
        st.caption(f"Searched within: {result['quality_group']}")
        st.markdown(
            f"### {wine['vintage']} {wine['producer']} {wine['varietal']}"
        )
        st.write(f"**Region:** {wine['region']}")
        st.write(f"**Quality:** {quality_label(int(wine['quality_score']))}")
        st.write(
            f"📍 **Cellar Location:** {wine.get('rack_location') or 'Unassigned'}"
        )
        st.write(f"🍾 **Bottles Remaining:** {wine['quantity']}")

        st.markdown("#### Why this bottle")
        st.write(result["reason"])

        st.markdown("#### Serving Recommendation")
        st.write(f"🌡️ **Temperature:** {result['temperature']}")
        decant_minutes = int(wine.get("decant_minutes") or 0)
        if decant_minutes == 0:
            st.write("⏱️ **Decant:** No decant recommended")
        else:
            st.write(f"⏱️ **Decant:** {decant_minutes} minutes")


# -----------------------------------------------------------------------------
# TAB 3: MANAGE CELLAR
# -----------------------------------------------------------------------------
with tab3:
    st.subheader("Manage Cellar")
    st.write("Log tasting notes and update quantities after opening a bottle.")

    try:
        response = (
            supabase.table("wines")
            .select("*")
            .eq("is_consumed", False)
            .gt("quantity", 0)
            .order("producer")
            .execute()
        )
        wines = response.data

        if not wines:
            st.info("No active bottles to manage.")
        else:
            for w in wines:
                with st.expander(
                    f"{w['vintage']} {w['producer']} {w['varietal']} "
                    f"({w['quantity']} available)"
                ):
                    st.write(
                        f"**Quality:** {quality_label(int(w.get('quality_score') or 0))}"
                    )
                    st.write(
                        f"**Location:** {w.get('rack_location') or 'Unassigned'} | "
                        f"**Window:** {w['drink_window']}"
                    )
                    decant_minutes = int(w.get("decant_minutes") or 0)
                    st.write(
                        "**Decant:** No decant recommended"
                        if decant_minutes == 0
                        else f"**Decant:** {decant_minutes} minutes"
                    )

                    notes = st.text_area(
                        "Tasting Notes",
                        key=f"notes_{w['wine_id']}",
                        placeholder="How did it drink and pair?",
                    )

                    if st.button(
                        "Mark One Bottle Consumed",
                        key=f"btn_{w['wine_id']}",
                    ):
                        new_quantity = int(w["quantity"]) - 1
                        supabase.table("wines").update(
                            {
                                "quantity": new_quantity,
                                "is_consumed": new_quantity == 0,
                                "post_drink_notes": notes or None,
                            }
                        ).eq("wine_id", w["wine_id"]).execute()

                        paired = st.session_state.get("pairing_result")
                        if paired and int(paired["wine"]["wine_id"]) == int(
                            w["wine_id"]
                        ):
                            st.session_state.pop("pairing_result", None)

                        st.success("Cellar updated!")
                        st.rerun()

    except Exception as exc:
        st.error("The cellar could not be loaded.")
        st.code(f"{type(exc).__name__}: {exc}")
