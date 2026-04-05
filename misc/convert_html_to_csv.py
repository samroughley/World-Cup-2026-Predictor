""" 
Currently have the html file for the table of third place mappings. 
Want to convert to a csv formatted for later convenience.
"""

# Import packages 
import pandas as pd


# Load the html file
table = pd.read_html('third_place_table.html')[0]
table = table.fillna("")

third_place_strings = []
for _, row in table.iterrows():
    third_place_str = ""
    for i in range(12):
        if i == 0:
            third_place_str += row["Third-placed teams advance from groupsvte"]
        else:
            third_place_str += row[f"Third-placed teams advance from groupsvte.{i}"]


    third_place_strings.append(third_place_str)

table["Third-placed teams advanced"] = third_place_strings
table = table[["Third-placed teams advanced","1A vs","1B vs","1D vs","1E vs","1G vs","1I vs","1K vs","1L vs"]]
table = table.rename(columns={
    "1A vs": "1A",
    "1B vs": "1B",
    "1D vs": "1D",
    "1E vs": "1E",
    "1G vs": "1G",
    "1I vs": "1I",
    "1K vs": "1K",
    "1L vs": "1L"
})
table.to_csv("third_place_table.csv",index=False)

