import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load artifacts
with open('transformer.pkl', 'rb') as file:
    pt = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('final_model_TS.pkl', 'rb') as file:
    model_lr_t = pickle.load(file)

with open('final_model_Px.pkl', 'rb') as file:
    model_xgb = pickle.load(file)

with open('brand_price_map.pkl', 'rb') as file:
    brand_price_map = pickle.load(file)

with open('case_material_encoder.pkl', 'rb') as file:
    case_material_encoder = pickle.load(file)

with open('bracelet_material_encoder.pkl', 'rb') as file:
    bracelet_material_encoder = pickle.load(file)

with open('condition_mapper.pkl', 'rb') as file:
    condition_mapper = pickle.load(file)

# Custom CSS 
st.markdown("""
<style>
    .main {background: #fafafa;}
    h1 {color: #1a237e; border-bottom: 2px solid #1a237e;}
    .stSelectbox, .stNumberInput, .stSlider {padding: 10px; border-radius: 5px;}
    .pred-box {padding: 20px; border-radius: 10px; margin: 15px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    .trusted {background: #e8f5e9; border: 2px solid #43a047;}
    .not-trusted {background: #ffebee; border: 2px solid #e53935;}
    .price-pred {background: #e3f2fd; border: 2px solid #1e88e5;}
</style>
""", unsafe_allow_html=True)

def main():
    st.title("Luxury Watch Analytics Platform ⌚")
    st.markdown("Predict Price & Seller Trustworthiness")
    
    with st.form("watch_inputs"):
        # Watch Specifications Section
        st.header("Watch Details")
        col1, col2 = st.columns(2)
        
        with col1:
            brand = st.selectbox("Brand", list(brand_price_map.keys()))
            movement = st.selectbox("Movement Type", ['Automatic', 'Quartz', 'Manual winding'])
            case_material = st.selectbox("Case Material", ['Steel', 'Titanium', 'Rose gold', 'Yellow gold', 'Gold/Steel'])
            bracelet_material = st.selectbox("Bracelet Material", ['Leather', 'Steel', 'Rubber', 'Crocodile skin', 'Textile'])
            year_production = st.slider("Production Year", 1900, 2024, 2020)
            water_resistance = st.slider("Water Resistance (meters)", 1, 200, 50)
            crystal = st.selectbox("Crystal Material", ['Sapphire crystal', 'Mineral Glass', 'Plexiglass'])
            
        with col2:
            condition = st.selectbox("Condition", ['New', 'Like new & unworn', 'Used (Very good)', 'Used (Good)', 'Used (Fair)'])
            scope_delivery = st.selectbox("Package Includes", ['Original box, original papers', 'No original box, no original papers'])
            gender = st.selectbox("Target Gender", ["Men's watch/Unisex", "Women's watch", "Unisex"])
            shape = st.selectbox("Watch Shape", ['Circular', 'Rectangular'])
            face_area = st.slider("Face Area (mm²)", 300, 2000, 600)
        
        # Seller Information Section
        st.header("Seller Information")
        col3, col4 = st.columns(2)
        
        with col3:
            watches_sold = st.number_input("Total Watches Sold", min_value=0, value=100)
            active_listings = st.number_input("Active Listings", min_value=0, value=50)
            
        with col4:
            fast_shipper = st.radio("Fast Shipping", [1, 0], format_func=lambda x: "Yes" if x else "No")
            punctuality = st.radio("Punctuality Rating", [1, 0], format_func=lambda x: "High" if x else "Low")
            seller_reviews = st.number_input("Seller Reviews Count", min_value=0, value=500)

        submitted = st.form_submit_button("Generate Predictions")
    
    if submitted:
        try:
            # Create input data (Price NOT included)
            input_data = {
                'Brand': brand_price_map[brand],
                'Movement': 0 if movement == 'Automatic' else 1,
                'Case material': case_material_encoder[case_material],
                'Bracelet material': bracelet_material_encoder[bracelet_material],
                'Year of production': year_production,
                'Condition': condition_mapper[condition],
                'Scope of delivery': 2 if 'Original box' in scope_delivery else 0,
                'Gender': 2 if "Men's" in gender else 0,
                'Availability': 2,
                'Shape': 0 if shape == 'Circular' else 1,
                'Water resistance': water_resistance,
                'Crystal': {'Sapphire crystal':4, 'Mineral Glass':3, 'Plexiglass':2}[crystal],
                'Dial': 0.33,
                'Bracelet color': 0.37,
                'Clasp': 5,
                'Watches Sold by the Seller': watches_sold,
                'Active listing of the seller': active_listings,
                'Fast Shipper': fast_shipper,
                'Punctuality': punctuality,
                'Seller Reviews': seller_reviews,
                'Face Area': face_area
            }
            
            # Convert to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Scale numerical features
            numerical_features = ['Watches Sold by the Seller', 'Active listing of the seller', 'Seller Reviews', 'Face Area']
            input_df[numerical_features] = scaler.transform(input_df[numerical_features])
            
            # Ensure feature order matches model
            input_df = input_df[model_xgb.feature_names_in_]
            
            # Make predictions
            price_pred = pt.inverse_transform(model_xgb.predict(input_df).reshape(-1, 1))[0][0]
            trust_prob = model_lr_t.predict_proba(input_df)[0][1]
            trust_status = model_lr_t.predict(input_df)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            col5, col6 = st.columns(2)
            with col5:
                st.markdown(f"""
                <div class="pred-box price-pred">
                    <h3>Predicted Value</h3>
                    <h2>${price_pred:,.2f}</h2>
                    <p>Estimated market price</p>
                </div>
                """, unsafe_allow_html=True)
                
            with col6:
                status_class = "trusted" if trust_status == 1 else "not-trusted"
                status_text = "Trusted Seller ✅" if trust_status == 1 else "Not Trusted ❌"
                st.markdown(f"""
                <div class="pred-box {status_class}">
                    <h3>Seller Reliability</h3>
                    <h2>{status_text}</h2>
                    <p>Confidence: {trust_prob*100:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
            st.write("Debug Info:")
            st.write("Input Features:", input_df.columns.tolist())
            st.write("Expected Features:", model_xgb.feature_names_in_.tolist())

if __name__ == "__main__":
    main()
