import pandas as pd

print("Loading dataset...")

df = pd.read_csv("arcos_sampled_50k.csv")

#needed columns
df = df[[
    "TRANSACTION_DATE",
    "Revised_Company_Name",
    "REPORTER_NAME",
    "BUYER_NAME",
    "BUYER_STATE",
    "QUANTITY"
]]

# rename columns
df.columns = [
    "date",
    "manufacturer",
    "distributor",
    "retailer",
    "retailer_state",
    "quantity"
]

# clean data
df = df.dropna()
df["date"] = pd.to_datetime(df["date"])

# save
df.to_csv("clean_chain.csv", index=False)

print("Saved: clean_chain.csv")
print("Rows:", len(df))
