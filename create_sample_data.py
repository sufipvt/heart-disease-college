import pandas as pd

# Load your original dataset
df = pd.read_csv("Heart_disease_cleveland_new.csv")

# Select 50 records and remove the target column
incoming_data = df.drop(columns=["target"]).sample(
    n=50,
    random_state=42
)

# Save the sample as incoming data
incoming_data.to_csv("incoming_clinical_data.csv", index=False)

print("Sample incoming data created successfully!")
print("Number of records:", len(incoming_data))