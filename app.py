import io
import json

import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError
from supabase import create_client, Client
from PIL import Image


# Initialize API clients
genai_client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

supabase: Client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"],
)


class WineData(BaseModel):
    wine_type: str = Field(
        description="Red, White, Rose, or Sparkling"
    )
    producer: str = Field(
        description="The winery or estate shown on the label"
    )
    vintage: str = Field(
        description="The four-digit year, or NV for non-vintage"
    )
    varietal: str = Field(
        description="The grape variety or blend"
    )
    region: str = Field(
        description="The geographic wine region"
    )
    flavor_notes: str = Field(
        description="Three or four inferred flavor notes"
    )
    tannin_level: str = Field(
        description="Low, Medium, or High"
    )
    acidity: str = Field(
        description="Low, Medium, or High"
    )
    body_type: str = Field(
        description="Light, Medium, or Full"
    )
    drink_window: str = Field(
        description="For example, Drink Now or Hold until 2028"
    )
    quality_score: int = Field(
        ge=1,
        le=5,
        description=(
            "An inferred quality estimate from 1 to 5 based on the "
            "information visible on the label and general wine knowledge"
        ),
    )


def analyze_wine_label(picture) -> dict:
    picture.seek(0)
    img = Image.open(picture).convert("RGB")

    img_byte_arr = io.BytesIO()
    img.save(
        img_byte_arr,
        format="JPEG",
        quality=90,
        optimize=True,
    )

    prompt = """
    Examine this wine label.

    Extract the wine type, producer, vintage, varietal, and region.
    Then infer likely flavor notes, tannin level, acidity, body type,
    ideal drink window, and an estimated quality score from 1 to 5.

    Use NV if no vintage is visible. Do not invent label details that
    cannot be read. When structural characteristics are not printed on
    the label, infer them from the producer, grape, region, and style.
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

    validated = WineData.model_validate_json(response.text)
    return validated.model_dump()


st.title("🍷 Pocket Sommelier")

tab1, tab2, tab3 = st.tabs(
    ["Scan & Add", "Sommelier Pairing", "Manage Cellar"]
)


with tab1:
    st.write(
        "Scan a label to automatically extract and infer its profile."
    )

    picture = st.camera_input("Scan Wine Label")

    if picture is not None:
        if st.button("Analyze Label", type="primary"):
            try:
                with st.spinner(
                    "Analyzing label and inferring structural profile..."
                ):
                    st.session_state["wine_data"] = analyze_wine_label(
                        picture
                    )

            except ValidationError as exc:
                st.error(
                    "Gemini returned JSON that did not match the wine schema."
                )
                st.code(str(exc))

            except Exception as exc:
                st.error("The wine label could not be analyzed.")
                st.code(f"{type(exc).__name__}: {exc}")

    data = st.session_state.get("wine_data")

    if data:
        st.success(
            f"Identified: {data['vintage']} "
            f"{data['producer']} {data['varietal']}"
        )

        st.json(data)

        with st.form("add_wine_form", clear_on_submit=True):
            st.subheader("Review and Add Logistics")

            col1, col2 = st.columns(2)

            with col1:
                rack_loc = st.text_input(
                    "Rack Location",
                    placeholder="e.g., Top row",
                )
                price = st.number_input(
                    "Purchase Price ($)",
                    min_value=0.0,
                    step=1.0,
                    value=None,
                )

            with col2:
                source = st.text_input(
                    "Source",
                    placeholder="e.g., Vin Rouge, Rue Cler",
                )

            submitted = st.form_submit_button("Save to Cellar")

            if submitted:
                record = {
                    **data,
                    "rack_location": rack_loc or None,
                    "source": source or None,
                    "purchase_price": price,
                    "quantity": 1,
                    "is_consumed": False,
                }

                try:
                    supabase.table("wines").insert(record).execute()
                    st.session_state.pop("wine_data", None)
                    st.success("Bottle added to inventory!")

                except Exception as exc:
                    st.error("The bottle could not be saved.")
                    st.code(f"{type(exc).__name__}: {exc}")
