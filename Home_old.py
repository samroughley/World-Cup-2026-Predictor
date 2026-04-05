import streamlit as st 
import pandas as pd 
import json

# Create page
st.set_page_config(layout="wide", page_title="WC 2026 Predictor")

st.title("🏆 2026 World Cup Probability Dashboard")



## Load Data ##

with open("Monte Carlo Simulation/runs/quick_run/final_results.json","r") as f:
    simulation_results = json.load(f)

opta_preds = pd.read_csv('misc/Opta_Predictions.csv')[['Team','R32_Exit','R16','QF','SF','Final','Winner']]
opta_preds[['R32_Exit','R16','QF','SF','Final','Winner']] = opta_preds[['R32_Exit','R16','QF','SF','Final','Winner']].astype('float')
print(opta_preds)


format_dict = {
    "R32_Exit": "{:.2f}%",
    "R16": "{:.2f}%",
    "QF": "{:.2f}%",
    "SF": "{:.2f}%",
    "Final": "{:.2f}%",
    "Winner": "{:.2f}%"
}
# A snippet of how to style it like Opta's site:
def style_probabilities(styler):
    # styler.background_gradient(axis=0, cmap="YlOrRd", vmin=0, vmax=20) # Red/Yellow heatmap
    styler.format(format_dict)
    return styler

# Display the table
# st.dataframe(opta_preds.style.pipe(style_probabilities), height=600, use_container_width=True, hide_index=True)
# st.table(opta_preds)


# Sample Data (Substitute with your actual scraped/simulated data)
data = {
    "Team": ["Spain", "France", "Senegal"],
    "Winner_Mine": [18.2, 11.5, 1.2],
    "Winner_Opta": [15.83, 12.77, 0.84]
}
df = pd.DataFrame(data)

# Create the combined display column
def combine_probs(row, my_col, opta_col):
    return f"{row[my_col]:.2f}% <span class='opta-text'>({row[opta_col]:.2f}%)</span>"

# Apply this to your knockout columns
df["Winner_Display"] = df.apply(combine_probs, axis=1, args=("Winner_Mine", "Winner_Opta"))



# Custom CSS to make the bracketed text faint
st.markdown("""
    <style>
    .opta-text {
        color: #808080; /* Grey color */
        font-size: 0.85em;
        font-weight: 300;
    }
    table {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🏆 WC 2026: My Model vs. Opta")
st.write("Main value is my prediction; bracketed value is Opta's.")

# Use st.write with the HTML-enabled dataframe
# Note: We only show the Display columns
display_df = df[["Team", "Winner_Display"]]

# Convert to HTML to preserve the <span> tags
st.write(display_df.to_html(escape=False, index=False), unsafe_allow_html=True)