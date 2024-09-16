import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go

# Custom CSS to beautify the app with the new color palette
st.markdown("""
<style>
    .main {
        background-color: #132A13;  /* Main background color */
        color: #ecf39e;
    }
    .stButton>button {
        color: #ecf39e;
        background-color: #31572c;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease 0s;
    }
    .stButton>button:hover {
        background-color: #4f772d;
        box-shadow: 0px 15px 20px rgba(79, 119, 45, 0.4);
        transform: translateY(-7px);
    }
    .stTextInput>div>div>input {
        color: #31572c;
        border-color: #90a955;
    }
    .stSelectbox>div>div>select {
        color: #31572c;
        border-color: #90a955;
    }
    .stSlider>div>div>div>div {
        color: #31572c;
    }
    h1 {
        color: #ecf39e;
        text-align: center;
        animation: fadeIn 1.5s;
        text-shadow: 2px 2px 4px rgba(144, 169, 85, 0.5);
    }
    h2 {
        color: #ecf39e;
        animation: slideIn 1s;
    }
    h3 {
        color: #4f772d;
    }
    p {
        color: #ecf39e;
    }
    .sidebar .sidebar-content {
        background-color: #132A13;  /* Sidebar background color */
        color: #ecf39e;
    }
    .footer {
        background-color: #31572c;
        color: #ecf39e;
        text-align: center;
        padding: 10px;
        font-style: italic;
    }
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    @keyframes slideIn {
        0% {transform: translateX(-100%);}
        100% {transform: translateX(0);}
    }
</style>
""", unsafe_allow_html=True)

# Language options for the app
languages = {"English": "en", "Hindi": "hi", "Kannada": "kn"}
selected_language = st.sidebar.selectbox("🌐 Select Language", list(languages.keys()), index=0)
language = languages[selected_language]

# Translations for multi-language support
translations = {
    "en": {
        "title": "🌱 Smart Fertilizer Recommendation System",
        "instructions": "How to Use This App",
        "steps": [
            "Input your farm's environmental conditions and current fertilizer levels using the sidebar.",
            "Select your soil type and crop type from the dropdown menus.",
            "Enter your land area to get personalized recommendations.",
            "Click the 'Recommend Fertilizer' button to get your optimized fertilizer suggestion.",
            "Review the recommendation, including fertilizer type, amount, and NPK distribution."
        ],
        "input_parameters": "📊 Input Parameters",
        "recommend_fertilizer_button": "Recommend Fertilizer",
        "fertilizer_type": "🌿 Recommended Fertilizer",
        "fertilizer_type_label": "Fertilizer Type",
        "amount_label": "Recommended Amount",
        "optimized_label": "Optimized Amount per Unit Area",
        "npk_distribution": "📊 Suggested NPK Distribution",
        "footer": "Stay rooted. Your plants will thank you! 🌱"
    },
    "hi": {
        "title": "🌱 स्मार्ट उर्वरक सिफारिश प्रणाली",
        "instructions": "इस ऐप का उपयोग कैसे करें",
        "steps": [
            "साइडबार का उपयोग करके अपने खेत की पर्यावरणीय स्थितियों और वर्तमान उर्वरक स्तरों को दर्ज करें।",
            "ड्रॉपडाउन मेनू से अपनी मिट्टी का प्रकार और फसल का प्रकार चुनें।",
            "व्यक्तिगत सिफारिशें प्राप्त करने के लिए अपने भूमि क्षेत्र को दर्ज करें।",
            "'उर्वरक सिफारिश' बटन पर क्लिक करें और अनुकूलित उर्वरक सुझाव प्राप्त करें।",
            "सिफारिश की समीक्षा करें, जिसमें उर्वरक प्रकार, मात्रा और एनपीके वितरण शामिल है।"
        ],
        "input_parameters": "📊 इनपुट पैरामीटर",
        "recommend_fertilizer_button": "उर्वरक सिफारिश करें",
        "fertilizer_type": "🌿 अनुशंसित उर्वरक",
        "fertilizer_type_label": "उर्वरक प्रकार",
        "amount_label": "अनुशंसित मात्रा",
        "optimized_label": "प्रति इकाई क्षेत्र में अनुकूलित मात्रा",
        "npk_distribution": "📊 सुझाया गया एनपीके वितरण",
        "footer": "जड़ें गहरी रखें। आपके पौधे आपको धन्यवाद देंगे! 🌱"
    },
    "kn": {
        "title": "🌱 ಸ್ಮಾರ್ಟ್ ರಸಗೊಬ್ಬರ ಶಿಫಾರಸು ವ್ಯವಸ್ಥೆ",
        "instructions": "ಈ ಅಪ್ಲಿಕೇಶನ್ ಅನ್ನು ಹೇಗೆ ಬಳಸುವುದು",
        "steps": [
            "ಸೈಡ್ಬಾರ್ ಬಳಸಿ ನಿಮ್ಮ ತೋಟದ ಪರಿಸರದ ಸ್ಥಿತಿಗಳನ್ನು ಮತ್ತು ಪ್ರಸ್ತುತ ರಸಗೊಬ್ಬರದ ಮಟ್ಟವನ್ನು ನಮೂದಿಸಿ.",
            "ಡ್ರಾಪ್‌ಡೌನ್ ಮೆನುಗಳಿಂದ ನಿಮ್ಮ ಮಣ್ಣಿನ ಪ್ರಕಾರ ಮತ್ತು ಬೆಳೆ ಪ್ರಕಾರವನ್ನು ಆರಿಸಿ.",
            "ವೈಯಕ್ತಿಕ ಶಿಫಾರಸುಗಳನ್ನು ಪಡೆಯಲು ನಿಮ್ಮ ಭೂಮಿ ಪ್ರದೇಶವನ್ನು ನಮೂದಿಸಿ.",
            "'ರಸಗೊಬ್ಬರ ಶಿಫಾರಸು' ಬಟನ್ ಕ್ಲಿಕ್ ಮಾಡಿ ಮತ್ತು ಅನುಕೂಲಿತ ರಸಗೊಬ್ಬರ ಶಿಫಾರಸು ಪಡೆಯಿರಿ.",
            "ಶಿಫಾರಸನ್ನು ಪರಿಶೀಲಿಸಿ, ರಸಗೊಬ್ಬರದ ಪ್ರಕಾರ, ಪ್ರಮಾಣ ಮತ್ತು ಎನ್‌ಪಿಕೆ ವಿತರಣೆ."
        ],
        "input_parameters": "📊 ಇನ್ಪುಟ್ ಪ್ಯಾರಾಮೀಟರ್‌ಗಳು",
        "recommend_fertilizer_button": "ರಸಗೊಬ್ಬರ ಶಿಫಾರಸು ಮಾಡಿ",
        "fertilizer_type": "🌿 ಶಿಫಾರಸು ಮಾಡಲಾದ ರಸಗೊಬ್ಬರ",
        "fertilizer_type_label": "ರಸಗೊಬ್ಬರದ ಪ್ರಕಾರ",
        "amount_label": "ಶಿಫಾರಸು ಮಾಡಿದ ಪ್ರಮಾಣ",
        "optimized_label": "ಪ್ರತಿ ಘಟಕದ ಪ್ರದೇಶಕ್ಕೆ ಪರಿಪೂರ್ಣ ಪ್ರಮಾಣ",
        "npk_distribution": "📊 ಶಿಫಾರಸು ಮಾಡಲಾದ ಎನ್‌ಪಿಕೆ ವಿತರಣೆ",
        "footer": "ಬೇರುಗಳನ್ನು ಗಟ್ಟಿಯಾಗಿ ಇರಿಸಿ. ನಿಮ್ಮ ಸಸ್ಯಗಳು ನಿಮಗೆ ಧನ್ಯವಾದ ಹೇಳುತ್ತವೆ! 🌱"
    }
}

t = translations[language]

st.title(t["title"])

st.markdown(f"""
    <div style='background-color: #90a955; padding: 15px; border-radius: 5px; animation: fadeIn 1.5s;'>
        <h2 style='color: #ecf39e;'>{t["instructions"]}</h2>
        <ol style='color: #ecf39e;'>
            <li>{t["steps"][0]}</li>
            <li>{t["steps"][1]}</li>
            <li>{t["steps"][2]}</li>
            <li>{t["steps"][3]}</li>
            <li>{t["steps"][4]}</li>
        </ol>
    </div>
""", unsafe_allow_html=True)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Fertilizer Prediction.csv")
    
    soil_type_encoder = LabelEncoder()
    crop_type_encoder = LabelEncoder()
    fertilizer_encoder = LabelEncoder()
    
    df['Soil Type'] = soil_type_encoder.fit_transform(df['Soil Type'])
    df['Crop Type'] = crop_type_encoder.fit_transform(df['Crop Type'])
    df['Fertilizer_Name'] = fertilizer_encoder.fit_transform(df['Fertilizer_Name'])
    
    df['Total_Fertilizer'] = df['Nitrogen'] + df['Potassium'] + df['Phosphorous']
    
    return df, soil_type_encoder, crop_type_encoder, fertilizer_encoder

df, soil_type_encoder, crop_type_encoder, fertilizer_encoder = load_data()

X = df.drop(['Fertilizer_Name', 'Total_Fertilizer'], axis=1)
y_type = df['Fertilizer_Name']
y_amount = df['Total_Fertilizer']

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X, y_type)

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X, y_amount)

def recommend_fertilizer(temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous, land_area=1):
    soil_type_encoded = soil_type_encoder.transform([soil_type])[0]

    if crop_type in crop_type_encoder.classes_:
        crop_type_encoded = crop_type_encoder.transform([crop_type])[0]
    else:
        st.warning(f"Crop type '{crop_type}' is not recognized. Defaulting to the first known crop type.")
        crop_type_encoded = crop_type_encoder.transform([crop_type_encoder.classes_[0]])[0]
    
    input_data = np.array([[temperature, humidity, moisture, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]])
    
    fertilizer_type = rf_classifier.predict(input_data)
    fertilizer_amount = rf_regressor.predict(input_data)
    
    recommended_fertilizer = fertilizer_encoder.inverse_transform(fertilizer_type)[0]
    optimized_amount = fertilizer_amount[0] / land_area
    
    return recommended_fertilizer, optimized_amount

# Sidebar for input features
st.sidebar.header(t["input_parameters"])

temperature = st.sidebar.slider('Temperature (°C)', 0, 50, 30)
humidity = st.sidebar.slider('Humidity (%)', 0, 100, 60)
moisture = st.sidebar.slider('Moisture (%)', 0, 100, 40)

soil_type = st.sidebar.selectbox('Soil Type', ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey'])

crop_type_options = crop_type_encoder.classes_
crop_type = st.sidebar.selectbox('Crop Type', crop_type_options)

nitrogen = st.sidebar.number_input('Current Nitrogen Amount', min_value=0.0, max_value=100.0, value=20.0)
phosphorous = st.sidebar.number_input('Current Phosphorous Amount', min_value=0.0, max_value=100.0, value=10.0)
potassium = st.sidebar.number_input('Current Potassium Amount', min_value=0.0, max_value=100.0, value=15.0)

land_area = st.sidebar.number_input('Land Area (units)', min_value=0.1, max_value=100.0, value=1.0)

# Predict the fertilizer type and amount
if st.sidebar.button(t["recommend_fertilizer_button"]):
    recommended_fertilizer, recommended_amount = recommend_fertilizer(
        temperature, humidity, moisture, soil_type, crop_type, nitrogen, potassium, phosphorous, land_area)

    st.markdown(f"""
        <div style='background-color: #31572c; padding: 20px; border-radius: 10px; margin-top: 20px;'>
            <h2 style='color: #ecf39e;'>{t["fertilizer_type"]}</h2>
            <h3 style='color: #90a955;'>{t["fertilizer_type_label"]}: <span style='color: #ecf39e;'>{recommended_fertilizer}</span></h3>
            <h3 style='color: #90a955;'>{t["amount_label"]}: <span style='color: #ecf39e;'>{recommended_amount:.2f} units (total)</span></h3>
            <h3 style='color: #90a955;'>{t["optimized_label"]}: <span style='color: #ecf39e;'>{recommended_amount / land_area:.2f} units</span></h3>
        </div>
    """, unsafe_allow_html=True)

    current_total = nitrogen + potassium + phosphorous
    if recommended_amount < current_total:
        st.warning(f"Suggestion: Reduce total fertilizer by {current_total - recommended_amount:.2f} units")
    elif recommended_amount > current_total:
        st.success(f"Suggestion: Increase total fertilizer by {recommended_amount - current_total:.2f} units")
    else:
        st.info("Current fertilizer amount is optimal")

    if '-' in recommended_fertilizer:
        npk_ratio = recommended_fertilizer.split('-')
        if len(npk_ratio) == 3:
            n_ratio, p_ratio, k_ratio = map(int, npk_ratio)
            total_ratio = n_ratio + p_ratio + k_ratio
            n_amount = (n_ratio / total_ratio) * recommended_amount
            p_amount = (p_ratio / total_ratio) * recommended_amount
            k_amount = (k_ratio / total_ratio) * recommended_amount
            
            st.subheader(t["npk_distribution"])
            
            fig = go.Figure(data=[go.Pie(labels=['Nitrogen', 'Phosphorous', 'Potassium'], values=[n_amount, p_amount, k_amount], hole=.3)])
            st.plotly_chart(fig)

# Footer with a calming message
st.markdown(f"<div class='footer'>{t['footer']}</div>", unsafe_allow_html=True)