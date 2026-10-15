# 🍷 Pocket Sommelier

A lightweight, cloud-hosted web application to track wine inventory via label photos and recommend pairings based on active stock and dish preparation.

---

## 🛠️ Architecture & Tech Stack
* **Frontend UI:** [Streamlit](https://streamlit.io/) (Mobile-responsive web interface using `st.camera_input()`).
* **AI Extraction & Sommelier Engine:** [Google Gemini API](https://aistudio.google.com/) (`gemini-2.5-flash` with structured outputs).
* **Cloud Database:** [Supabase](https://supabase.com/) (PostgreSQL with Row Level Security enabled).
* **Hosting:** Streamlit Community Cloud (deployed directly from main GitHub branch).

---

## 📊 Database Schema (`wines` table)

| Column | Type | Description |
| :--- | :--- | :--- |
| `wine_id` | UUID | Primary Key (auto-generated) |
| `producer` | TEXT | Winery or estate name |
| `vintage` | TEXT | Harvest year or NV |
| `varietal` | TEXT | Grape varietal or blend type |
| `region` | TEXT | Geographic growing region |
| `flavor_notes` | TEXT | 3–4 inferred primary aromas/flavors |
| `tannin_level` | TEXT | Low / Medium / High |
| `acidity` | TEXT | Low / Medium / High |
| `body_type` | TEXT | Light / Medium / Full |
| `drink_window` | TEXT | Estimated peak drinking window |
| `rack_location` | TEXT | Physical shelf/cellar coordinate |
| `source` | TEXT | Purchasing origin/merchant |
| `purchase_price` | NUMERIC | Original cost |
| `quantity` | INTEGER | Active bottle count |
| `post_drink_notes`| TEXT | Tasting log and pairing feedback |
| `is_consumed` | BOOLEAN | Flagged `TRUE` when quantity reaches 0 |

---

## ⚙️ Environment Variables (Streamlit Secrets)

To run this application, the following keys must be defined in **Streamlit Advanced Settings > Secrets**:

```toml
GEMINI_API_KEY = "your_gemini_api_key"
SUPABASE_URL = "your_supabase_project_url"
SUPABASE_KEY = "your_supabase_anon_public_key"
