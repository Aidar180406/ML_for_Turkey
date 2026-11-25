import streamlit as st
import requests
from typing import List, Dict, Any

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Прогноз стоимости недвижимости", layout="centered")

st.title("🏙️ Прогноз стоимости недвижимости (İstanbul)")
st.caption("Модель: RandomForestRegressor, R²=0.595")

# --- Загрузка справочников ---
@st.cache_data(ttl=3600)
def get_metadata() -> Dict[str, Any]:
    try:
        resp = requests.get(f"{API_URL}/metadata", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}

metadata = get_metadata()

if not metadata:
    st.error("❌ Не удалось получить справочники. Проверьте, запущен ли API на http://localhost:8000")
    st.stop()

# --- Форма ввода ---

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    size = col1.number_input("Площадь, м²", min_value=20, max_value=1000, value=120, step=1.0)
    rooms_num = col2.number_input("Количество комнат", min_value=0, max_value=10, value=3, step=1)

    col3, col4 = st.columns(2)
    building_age_num = col3.number_input("Возраст здания, лет", min_value=0, max_value=100, value=5, step=1.0)
    total_floor_count_num = col4.number_input("Всего этажей", min_value=1, max_value=50, value=10, step=1.0)

    st.divider()

    sub_type = st.selectbox("Тип объекта", metadata.get("sub_types", ["Daire"]))
    listing_type = st.selectbox(
        "Тип предложения",
        options=[1, 2],
        format_func=lambda x: {1: "Продажа", 2: "Аренда"}.get(x, str(x)),
    )
    heating_type = st.selectbox("Отопление", metadata.get("heating_types", ["Kombi (Doğalgaz)"]))
    city = st.selectbox("Город", metadata.get("cities", ["İstanbul"]))
    floor_category = st.selectbox("Категория этажа", metadata.get("floor_categories", ["floor_3"]))

    # Кнопка — ВНУТРИ with, с тем же отступом, что и selectbox
    submitted = st.form_submit_button("Рассчитать прогноз")

if submitted:
    with st.spinner("Считаем прогноз..."):
        payload = {
            "size": float(size),
            "rooms_num": int(rooms_num),
            "building_age_num": float(building_age_num),
            "total_floor_count_num": float(total_floor_count_num),
            "sub_type": sub_type,
            "listing_type": int(listing_type),
            "heating_type": heating_type,
            "city": city,
            "floor_category": floor_category,
        }

        try:
            resp = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            st.success("✅ Прогноз готов!")

            col1, col2 = st.columns([3, 1])
            col1.metric(
                label="Прогнозируемая цена",
                value=f"{result['prediction']:,.0f} TRY",
                delta=f"±{result['mae']:,.0f} TRY"
            )
            col2.metric("R² модели", f"{result['r2_score']:.3f}")

            st.info(f"📊 Интервал цены: от {result['min_price']:,.0f} до {result['max_price']:,.0f} TRY ({result['confidence_interval']})")
            st.markdown(f"**MAPE модели:** {result['mape']:.2f}% — средняя ошибка в процентах.")

        except requests.exceptions.RequestException as e:
            st.error(f"❌ Ошибка связи с API: {e}")
        except Exception as e:
            st.error(f"❌ Неожиданная ошибка: {e}")
