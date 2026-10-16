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
    wine_type: str = Field(description="Red, White, Rose, Sparkling or Dessert")
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
    wine_id: str = Field(description="The exact wine_id of the selected wine")
    reason: str = Field(description="Why this specific wine pairs with the request")
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
    ["Scan & Add", "Find a Wine", "Manage Cellar"]
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
                wine_types = [
                    "Red",
                    "White",
                    "Rose",
                    "Sparkling",
                    "Dessert",
                ]
                
                wine_type = st.selectbox(
                    "Wine Type",
                    wine_types,
                    index=(
                        wine_types.index(data["wine_type"])
                        if data["wine_type"] in wine_types
                        else 0
                    ),
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
# TAB 2: FIND A WINE
# -----------------------------------------------------------------------------
with tab2:
    st.subheader("🍷 Find a Wine")
    st.write(
        "Search Supabase first using strict cellar filters. Gemini is only called "
        "if you explicitly ask the sommelier to choose among the matching bottles."
    )

    request_text = st.text_area(
        "Optional preference or meal",
        placeholder=(
            "Examples: good acidity; medium-bodied and not too tannic; sushi; "
            "ready to drink tonight"
        ),
        key="wine_request",
    )

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        quality_group = st.selectbox(
            "Quality",
            [
                "Any quality",
                "3 - Good / regular drinking",
                "4 - Excellent",
                "5 - Exceptional / special occasion",
                "1-2 - Lower-rated bottles",
            ],
        )
    with filter_col2:
        wine_type_filter = st.selectbox(
            "Wine type",
            ["Any type", "Red", "White", "Rose", "Sparkling"],
        )

    quality_map = {
        "1-2 - Lower-rated bottles": [1, 2],
        "3 - Good / regular drinking": [3],
        "4 - Excellent": [4],
        "5 - Exceptional / special occasion": [5],
    }

    if st.button("Search My Cellar", type="primary"):
        try:
            query = (
                supabase.table("wines")
                .select("*")
                .eq("is_consumed", False)
                .gt("quantity", 0)
            )

            if quality_group != "Any quality":
                query = query.in_("quality_score", quality_map[quality_group])
            if wine_type_filter != "Any type":
                query = query.eq("wine_type", wine_type_filter)

            response = query.execute()
            matches = response.data

            # Apply exact structured preferences locally without using Gemini.
            preference = request_text.strip().lower()
            if preference:
                if "high acidity" in preference or "good acidity" in preference:
                    matches = [w for w in matches if w.get("acidity") == "High"]
                elif "medium acidity" in preference:
                    matches = [w for w in matches if w.get("acidity") == "Medium"]
                elif "low acidity" in preference:
                    matches = [w for w in matches if w.get("acidity") == "Low"]

                if "full-bodied" in preference or "full bodied" in preference:
                    matches = [w for w in matches if w.get("body_type") == "Full"]
                elif "medium-bodied" in preference or "medium bodied" in preference:
                    matches = [w for w in matches if w.get("body_type") == "Medium"]
                elif "light-bodied" in preference or "light bodied" in preference:
                    matches = [w for w in matches if w.get("body_type") == "Light"]

                if "high tannin" in preference:
                    matches = [w for w in matches if w.get("tannin_level") == "High"]
                elif "medium tannin" in preference:
                    matches = [w for w in matches if w.get("tannin_level") == "Medium"]
                elif "low tannin" in preference or "not too tannic" in preference:
                    matches = [w for w in matches if w.get("tannin_level") == "Low"]

            matches = sorted(
                matches,
                key=lambda w: (
                    -(int(w.get("quality_score") or 0)),
                    str(w.get("producer") or ""),
                    str(w.get("vintage") or ""),
                ),
            )
            st.session_state["cellar_matches"] = matches
            st.session_state["cellar_search_request"] = request_text.strip()
            st.session_state["cellar_search_quality"] = quality_group
            st.session_state["cellar_search_type"] = wine_type_filter
            st.session_state.pop("pairing_result", None)

        except Exception as exc:
            st.error("The cellar search failed.")
            st.code(f"{type(exc).__name__}: {exc}")

    matches = st.session_state.get("cellar_matches")
    if matches is not None:
        if not matches:
            st.warning(
                "No available bottles matched those filters. Try broadening the "
                "quality, wine type, or structured preference."
            )
        else:
            st.markdown("---")
            st.subheader(f"Matching Bottles ({len(matches)})")
            st.caption("These results came directly from Supabase. No Gemini call was used.")

            for w in matches:
                with st.expander(
                    f"{w['vintage']} {w['producer']} {w['varietal']} "
                    f"| Quality {w['quality_score']}"
                ):
                    st.write(f"**Type:** {w['wine_type']} | **Region:** {w['region']}")
                    st.write(
                        f"**Profile:** {w['body_type']} body | "
                        f"{w['acidity']} acidity | {w['tannin_level']} tannin"
                    )
                    st.write(f"**Notes:** {w['flavor_notes']}")
                    st.write(f"**Drink window:** {w['drink_window']}")
                    st.write(
                        f"**Location:** {w.get('rack_location') or 'Unassigned'} | "
                        f"**Quantity:** {w['quantity']}"
                    )
                    minutes = int(w.get("decant_minutes") or 0)
                    st.write(
                        "**Decant:** None recommended"
                        if minutes == 0
                        else f"**Decant:** {minutes} minutes"
                    )

            if len(matches) == 1:
                st.info("Only one bottle matched, so an AI recommendation is unnecessary.")
            elif st.button("Ask Sommelier to Choose One"):
                try:
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
                                f"Drink window: {w['drink_window']}"
                            )
                            for w in matches
                        ]
                    )
                    user_request = (
                        st.session_state.get("cellar_search_request")
                        or "Choose the most appropriate bottle for drinking now."
                    )
                    prompt = f"""
                    Choose one wine from the filtered cellar results.

                    USER PREFERENCE OR MEAL:
                    {user_request}

                    AVAILABLE WINES:
                    {inventory_str}

                    Select one exact WINE ID from the list. Match food structure when
                    a meal is provided; otherwise match the requested acidity, tannin,
                    body, flavor, occasion, or readiness. Ignore price and prestige.
                    Explain briefly why it is the best match and give a serving temperature.
                    """
                    with st.spinner("Asking the sommelier to choose..."):
                        ai_response = genai_client.models.generate_content(
                            model="gemini-flash-latest",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=PairingRecommendation,
                                temperature=0.1,
                            ),
                        )
                    recommendation = PairingRecommendation.model_validate_json(
                        ai_response.text
                    )
                    selected = next(
                        (
                            w for w in matches
                            if str(w["wine_id"]) == str(recommendation.wine_id)
                        ),
                        None,
                    )
                    if selected is None:
                        st.error("The model selected a wine outside the search results.")
                    else:
                        st.session_state["pairing_result"] = {
                            "wine": selected,
                            "reason": recommendation.reason,
                            "temperature": recommendation.serving_temperature,
                        }
                        st.rerun()
                except ValidationError as exc:
                    st.error("The recommendation did not match the expected schema.")
                    st.code(str(exc))
                except Exception as exc:
                    st.error("The AI recommendation failed.")
                    st.code(f"{type(exc).__name__}: {exc}")

    if "pairing_result" in st.session_state:
        result = st.session_state["pairing_result"]
        wine = result["wine"]
        st.markdown("---")
        st.subheader("Sommelier's Choice")
        st.markdown(
            f"### {wine['vintage']} {wine['producer']} {wine['varietal']}"
        )
        st.write(result["reason"])
        st.write(f"🌡️ **Temperature:** {result['temperature']}")
        minutes = int(wine.get("decant_minutes") or 0)
        st.write(
            "⏱️ **Decant:** No decant recommended"
            if minutes == 0
            else f"⏱️ **Decant:** {minutes} minutes"
        )
        st.write(f"📍 **Location:** {wine.get('rack_location') or 'Unassigned'}")


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
                        if paired and str(paired["wine"]["wine_id"]) == str(
                            w["wine_id"]
                        ):
                            st.session_state.pop("pairing_result", None)

                        st.success("Cellar updated!")
                        st.rerun()

    except Exception as exc:
        st.error("The cellar could not be loaded.")
        st.code(f"{type(exc).__name__}: {exc}")
