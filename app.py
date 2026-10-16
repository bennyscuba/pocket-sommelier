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
        "Search your cellar using structured filters first. "
        "Gemini is used only when you ask the sommelier to choose "
        "among multiple matching bottles."
    )

    request_text = st.text_area(
        "Optional preference or meal",
        placeholder=(
            "Examples: good acidity; medium-bodied and not too tannic; "
            "something for sushi; ready to drink tonight"
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
            [
                "Any type",
                "Red",
                "White",
                "Rose",
                "Sparkling",
                "Dessert",
            ],
        )

    quality_map = {
        "1-2 - Lower-rated bottles": [1, 2],
        "3 - Good / regular drinking": [3],
        "4 - Excellent": [4],
        "5 - Exceptional / special occasion": [5],
    }

    # -------------------------------------------------------------------------
    # Search Supabase
    # -------------------------------------------------------------------------
    if st.button("Search My Cellar", type="primary"):
        try:
            query = (
                supabase.table("wines")
                .select("*")
                .eq("is_consumed", False)
                .gt("quantity", 0)
            )

            if quality_group != "Any quality":
                query = query.in_(
                    "quality_score",
                    quality_map[quality_group],
                )

            if wine_type_filter != "Any type":
                query = query.eq(
                    "wine_type",
                    wine_type_filter,
                )

            response = query.execute()
            matches = response.data or []

            # -------------------------------------------------------------
            # Apply simple structured text preferences locally.
            # These do not require Gemini.
            # -------------------------------------------------------------
            preference = request_text.strip().lower()

            if preference:
                # Acidity
                if (
                    "high acidity" in preference
                    or "good acidity" in preference
                    or "bright acidity" in preference
                    or "crisp" in preference
                ):
                    matches = [
                        w for w in matches
                        if str(w.get("acidity", "")).lower() == "high"
                    ]

                elif "medium acidity" in preference:
                    matches = [
                        w for w in matches
                        if str(w.get("acidity", "")).lower() == "medium"
                    ]

                elif "low acidity" in preference:
                    matches = [
                        w for w in matches
                        if str(w.get("acidity", "")).lower() == "low"
                    ]

                # Body
                if (
                    "full-bodied" in preference
                    or "full bodied" in preference
                    or "full body" in preference
                ):
                    matches = [
                        w for w in matches
                        if str(w.get("body_type", "")).lower() == "full"
                    ]

                elif (
                    "medium-bodied" in preference
                    or "medium bodied" in preference
                    or "medium body" in preference
                ):
                    matches = [
                        w for w in matches
                        if str(w.get("body_type", "")).lower() == "medium"
                    ]

                elif (
                    "light-bodied" in preference
                    or "light bodied" in preference
                    or "light body" in preference
                    or "lighter wine" in preference
                ):
                    matches = [
                        w for w in matches
                        if str(w.get("body_type", "")).lower() == "light"
                    ]

                # Tannin
                if (
                    "high tannin" in preference
                    or "high tannins" in preference
                    or "tannic" in preference
                ):
                    matches = [
                        w for w in matches
                        if str(w.get("tannin_level", "")).lower() == "high"
                    ]

                elif (
                    "medium tannin" in preference
                    or "medium tannins" in preference
                ):
                    matches = [
                        w for w in matches
                        if str(w.get("tannin_level", "")).lower() == "medium"
                    ]

                elif (
                    "low tannin" in preference
                    or "low tannins" in preference
                    or "not too tannic" in preference
                    or "soft tannins" in preference
                ):
                    matches = [
                        w for w in matches
                        if str(w.get("tannin_level", "")).lower() == "low"
                    ]

            # Sort without using purchase price.
            matches = sorted(
                matches,
                key=lambda w: (
                    -(int(w.get("quality_score") or 0)),
                    str(w.get("producer") or ""),
                    str(w.get("vintage") or ""),
                ),
            )

            st.session_state["cellar_matches"] = matches
            st.session_state["cellar_search_request"] = (
                request_text.strip()
            )
            st.session_state["cellar_search_quality"] = (
                quality_group
            )
            st.session_state["cellar_search_type"] = (
                wine_type_filter
            )

            # A new search starts a new recommendation sequence.
            st.session_state["rejected_wine_ids"] = []
            st.session_state.pop("pairing_result", None)

        except Exception as exc:
            st.error("The cellar search failed.")
            st.code(f"{type(exc).__name__}: {exc}")

    # -------------------------------------------------------------------------
    # Display direct Supabase matches
    # -------------------------------------------------------------------------
    matches = st.session_state.get("cellar_matches")

    if matches is not None:
        if not matches:
            st.warning(
                "No available bottles matched those filters. "
                "Try broadening the quality, wine type, or preference."
            )

        else:
            st.markdown("---")
            st.subheader(f"Matching Bottles ({len(matches)})")

            st.caption(
                "These results came directly from Supabase. "
                "No Gemini call was used."
            )

            for w in matches:
                wine_name = (
                    f"{w.get('vintage', 'NV')} "
                    f"{w.get('producer', 'Unknown Producer')} "
                    f"{w.get('varietal', '')}"
                )

                quality = w.get("quality_score")
                quantity = w.get("quantity", 0)

                with st.expander(
                    f"{wine_name} | Quality {quality}"
                ):
                    st.write(
                        f"**Type:** {w.get('wine_type', 'Unknown')} | "
                        f"**Region:** {w.get('region', 'Unknown')}"
                    )

                    st.write(
                        f"**Profile:** "
                        f"{w.get('body_type', 'Unknown')} body | "
                        f"{w.get('acidity', 'Unknown')} acidity | "
                        f"{w.get('tannin_level', 'Unknown')} tannin"
                    )

                    st.write(
                        f"**Notes:** "
                        f"{w.get('flavor_notes') or 'Not recorded'}"
                    )

                    st.write(
                        f"**Drink window:** "
                        f"{w.get('drink_window') or 'Not recorded'}"
                    )

                    st.write(
                        f"**Location:** "
                        f"{w.get('rack_location') or 'Unassigned'} | "
                        f"**Quantity:** {quantity}"
                    )

                    minutes = int(
                        w.get("decant_minutes") or 0
                    )

                    if minutes == 0:
                        st.write(
                            "**Decant:** None recommended"
                        )
                    else:
                        st.write(
                            f"**Decant:** {minutes} minutes"
                        )

            # -------------------------------------------------------------
            # If only one bottle matches, show it directly.
            # No Gemini call is necessary.
            # -------------------------------------------------------------
            if (
                len(matches) == 1
                and "pairing_result"
                not in st.session_state
            ):
                only_wine = matches[0]

                st.session_state["pairing_result"] = {
                    "wine": only_wine,
                    "reason": (
                        "This is the only available bottle matching "
                        "the selected cellar filters."
                    ),
                    "temperature": (
                        "Use the customary serving temperature "
                        "for this wine style."
                    ),
                    "selection_method": "Supabase filter",
                }

            # -------------------------------------------------------------
            # Ask Gemini to choose among multiple matches.
            # -------------------------------------------------------------
            elif (
                len(matches) > 1
                and "pairing_result"
                not in st.session_state
            ):
                if st.button(
                    "Ask Sommelier to Choose One",
                    type="primary",
                ):
                    try:
                        rejected_ids = set(
                            str(wine_id)
                            for wine_id in st.session_state.get(
                                "rejected_wine_ids",
                                [],
                            )
                        )

                        eligible_wines = [
                            w for w in matches
                            if str(w["wine_id"])
                            not in rejected_ids
                        ]

                        if not eligible_wines:
                            st.warning(
                                "There are no remaining bottles "
                                "to recommend."
                            )

                        elif len(eligible_wines) == 1:
                            selected = eligible_wines[0]

                            st.session_state[
                                "pairing_result"
                            ] = {
                                "wine": selected,
                                "reason": (
                                    "This is the only remaining "
                                    "bottle matching your search."
                                ),
                                "temperature": (
                                    "Use the customary serving "
                                    "temperature for this wine style."
                                ),
                                "selection_method": (
                                    "Last remaining match"
                                ),
                            }

                            st.rerun()

                        else:
                            inventory_str = "\n\n".join(
                                [
                                    (
                                        f"WINE ID: "
                                        f"{w['wine_id']}\n"
                                        f"Wine: "
                                        f"{w.get('vintage', 'NV')} "
                                        f"{w.get('producer', '')} "
                                        f"{w.get('varietal', '')}\n"
                                        f"Type: "
                                        f"{w.get('wine_type', '')}\n"
                                        f"Region: "
                                        f"{w.get('region', '')}\n"
                                        f"Quality score: "
                                        f"{w.get('quality_score', '')}\n"
                                        f"Tannins: "
                                        f"{w.get('tannin_level', '')}\n"
                                        f"Acidity: "
                                        f"{w.get('acidity', '')}\n"
                                        f"Body: "
                                        f"{w.get('body_type', '')}\n"
                                        f"Flavor notes: "
                                        f"{w.get('flavor_notes', '')}\n"
                                        f"Drink window: "
                                        f"{w.get('drink_window', '')}"
                                    )
                                    for w in eligible_wines
                                ]
                            )

                            user_request = (
                                st.session_state.get(
                                    "cellar_search_request"
                                )
                                or (
                                    "Choose the most appropriate "
                                    "bottle for drinking now."
                                )
                            )

                            prompt = f"""
Choose one wine from the filtered cellar results.

USER PREFERENCE OR MEAL:
{user_request}

AVAILABLE WINES:
{inventory_str}

Select one exact WINE ID from the list.

If the request describes a meal, choose based on food and
wine structure. Otherwise, match the requested acidity,
tannin, body, flavor, occasion, or drinking window.

Do not use price or monetary value when selecting the wine.
Do not choose a bottle outside the available wine list.

Explain briefly why the selected wine is the best match and
provide an appropriate serving temperature.
"""

                            with st.spinner(
                                "Asking the sommelier "
                                "to choose..."
                            ):
                                ai_response = (
                                    genai_client.models
                                    .generate_content(
                                        model=(
                                            "gemini-flash-latest"
                                        ),
                                        contents=prompt,
                                        config=(
                                            types
                                            .GenerateContentConfig(
                                                response_mime_type=(
                                                    "application/json"
                                                ),
                                                response_schema=(
                                                    PairingRecommendation
                                                ),
                                                temperature=0.1,
                                            )
                                        ),
                                    )
                                )

                            recommendation = (
                                PairingRecommendation
                                .model_validate_json(
                                    ai_response.text
                                )
                            )

                            # UUIDs must be compared as strings.
                            selected = next(
                                (
                                    w for w in eligible_wines
                                    if str(w["wine_id"])
                                    == str(
                                        recommendation.wine_id
                                    )
                                ),
                                None,
                            )

                            if selected is None:
                                st.error(
                                    "The model selected a wine "
                                    "outside the search results."
                                )

                            else:
                                st.session_state[
                                    "pairing_result"
                                ] = {
                                    "wine": selected,
                                    "reason": (
                                        recommendation.reason
                                    ),
                                    "temperature": (
                                        recommendation
                                        .serving_temperature
                                    ),
                                    "selection_method": (
                                        "Sommelier recommendation"
                                    ),
                                }

                                st.rerun()

                    except ValidationError as exc:
                        st.error(
                            "The recommendation did not "
                            "match the expected schema."
                        )
                        st.code(str(exc))

                    except Exception as exc:
                        st.error(
                            "The AI recommendation failed."
                        )
                        st.code(
                            f"{type(exc).__name__}: {exc}"
                        )

    # -------------------------------------------------------------------------
    # Display current recommendation
    # -------------------------------------------------------------------------
    if "pairing_result" in st.session_state:
        result = st.session_state["pairing_result"]
        wine = result["wine"]

        st.markdown("---")
        st.subheader("Sommelier's Choice")

        st.caption(
            f"Selection method: "
            f"{result.get('selection_method', 'Recommendation')}"
        )

        st.markdown(
            f"### {wine.get('vintage', 'NV')} "
            f"{wine.get('producer', 'Unknown Producer')} "
            f"{wine.get('varietal', '')}"
        )

        st.write(
            f"**Type:** {wine.get('wine_type', 'Unknown')}"
        )

        st.write(
            f"**Region:** {wine.get('region', 'Unknown')}"
        )

        st.write(
            f"**Quality:** "
            f"{quality_label(int(wine['quality_score']))}"
        )

        st.write(
            f"**Profile:** "
            f"{wine.get('body_type', 'Unknown')} body | "
            f"{wine.get('acidity', 'Unknown')} acidity | "
            f"{wine.get('tannin_level', 'Unknown')} tannin"
        )

        st.markdown("#### Why this bottle")
        st.write(result["reason"])

        st.markdown("#### Serving Recommendation")
        st.write(
            f"🌡️ **Temperature:** "
            f"{result['temperature']}"
        )

        minutes = int(
            wine.get("decant_minutes") or 0
        )

        if minutes == 0:
            st.write(
                "⏱️ **Decant:** No decant recommended"
            )
        else:
            st.write(
                f"⏱️ **Decant:** {minutes} minutes"
            )

        st.write(
            f"📍 **Location:** "
            f"{wine.get('rack_location') or 'Unassigned'}"
        )

        st.write(
            f"🍾 **Bottles Remaining:** "
            f"{wine.get('quantity', 0)}"
        )

        # Price is displayed after selection but is not sent to Gemini.
        purchase_price = wine.get("purchase_price")

        if purchase_price is None:
            st.write(
                "💰 **Purchase Price:** Not recorded"
            )
        else:
            try:
                formatted_price = float(purchase_price)
                st.write(
                    f"💰 **Purchase Price:** "
                    f"${formatted_price:,.2f}"
                )
            except (TypeError, ValueError):
                st.write(
                    f"💰 **Purchase Price:** "
                    f"{purchase_price}"
                )

        # ---------------------------------------------------------------------
        # Choose a different recommendation
        # ---------------------------------------------------------------------
        all_matches = st.session_state.get(
            "cellar_matches",
            [],
        )

        rejected_ids = set(
            str(wine_id)
            for wine_id in st.session_state.get(
                "rejected_wine_ids",
                [],
            )
        )

        current_id = str(wine["wine_id"])

        remaining_wines = [
            w for w in all_matches
            if str(w["wine_id"]) != current_id
            and str(w["wine_id"]) not in rejected_ids
        ]

        if remaining_wines:
            if st.button(
                "🔄 Choose a Different One",
                type="secondary",
            ):
                try:
                    # Record the current recommendation as rejected.
                    rejected_list = (
                        st.session_state.get(
                            "rejected_wine_ids",
                            [],
                        )
                    )

                    if current_id not in [
                        str(item)
                        for item in rejected_list
                    ]:
                        rejected_list.append(current_id)

                    st.session_state[
                        "rejected_wine_ids"
                    ] = rejected_list

                    updated_remaining = [
                        w for w in all_matches
                        if str(w["wine_id"])
                        not in set(
                            str(item)
                            for item in rejected_list
                        )
                    ]

                    # If only one bottle remains, choose it without Gemini.
                    if len(updated_remaining) == 1:
                        new_selection = updated_remaining[0]

                        st.session_state[
                            "pairing_result"
                        ] = {
                            "wine": new_selection,
                            "reason": (
                                "This is the only remaining "
                                "bottle matching your search."
                            ),
                            "temperature": (
                                "Use the customary serving "
                                "temperature for this wine style."
                            ),
                            "selection_method": (
                                "Last remaining match"
                            ),
                        }

                        st.rerun()

                    elif len(updated_remaining) > 1:
                        inventory_str = "\n\n".join(
                            [
                                (
                                    f"WINE ID: "
                                    f"{w['wine_id']}\n"
                                    f"Wine: "
                                    f"{w.get('vintage', 'NV')} "
                                    f"{w.get('producer', '')} "
                                    f"{w.get('varietal', '')}\n"
                                    f"Type: "
                                    f"{w.get('wine_type', '')}\n"
                                    f"Region: "
                                    f"{w.get('region', '')}\n"
                                    f"Quality score: "
                                    f"{w.get('quality_score', '')}\n"
                                    f"Tannins: "
                                    f"{w.get('tannin_level', '')}\n"
                                    f"Acidity: "
                                    f"{w.get('acidity', '')}\n"
                                    f"Body: "
                                    f"{w.get('body_type', '')}\n"
                                    f"Flavor notes: "
                                    f"{w.get('flavor_notes', '')}\n"
                                    f"Drink window: "
                                    f"{w.get('drink_window', '')}"
                                )
                                for w in updated_remaining
                            ]
                        )

                        user_request = (
                            st.session_state.get(
                                "cellar_search_request"
                            )
                            or (
                                "Choose the most appropriate "
                                "bottle for drinking now."
                            )
                        )

                        alternate_prompt = f"""
Choose a different wine from the remaining filtered cellar
results.

ORIGINAL USER PREFERENCE OR MEAL:
{user_request}

REMAINING AVAILABLE WINES:
{inventory_str}

The user declined the prior recommendation. Select one exact
WINE ID from the remaining list.

Choose the remaining wine that best satisfies the original
request. Do not use price or monetary value in the decision.
Do not select a bottle outside the remaining list.

Explain briefly why this alternative is a good match and
provide an appropriate serving temperature.
"""

                        with st.spinner(
                            "Choosing another bottle..."
                        ):
                            alternate_response = (
                                genai_client.models
                                .generate_content(
                                    model=(
                                        "gemini-flash-latest"
                                    ),
                                    contents=alternate_prompt,
                                    config=(
                                        types
                                        .GenerateContentConfig(
                                            response_mime_type=(
                                                "application/json"
                                            ),
                                            response_schema=(
                                                PairingRecommendation
                                            ),
                                            temperature=0.1,
                                        )
                                    ),
                                )
                            )

                        alternate = (
                            PairingRecommendation
                            .model_validate_json(
                                alternate_response.text
                            )
                        )

                        new_selection = next(
                            (
                                w
                                for w in updated_remaining
                                if str(w["wine_id"])
                                == str(alternate.wine_id)
                            ),
                            None,
                        )

                        if new_selection is None:
                            st.error(
                                "The model selected a wine "
                                "outside the remaining choices."
                            )

                        else:
                            st.session_state[
                                "pairing_result"
                            ] = {
                                "wine": new_selection,
                                "reason": alternate.reason,
                                "temperature": (
                                    alternate
                                    .serving_temperature
                                ),
                                "selection_method": (
                                    "Alternative recommendation"
                                ),
                            }

                            st.rerun()

                except ValidationError as exc:
                    st.error(
                        "The alternative recommendation did "
                        "not match the expected schema."
                    )
                    st.code(str(exc))

                except Exception as exc:
                    st.error(
                        "I couldn't choose another bottle."
                    )
                    st.code(
                        f"{type(exc).__name__}: {exc}"
                    )

        else:
            st.info(
                "There are no other matching bottles "
                "remaining in this search."
            )
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
