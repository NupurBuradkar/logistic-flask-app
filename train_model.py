import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

# Better dataset (clear pattern)
data = {
    "hours": [1,2,3,4,5,6,7,8,9,10],
    "attendance": [40,50,60,65,70,75,80,85,90,95],
    "marks": [20,30,40,45,50,60,70,80,85,90],
    "pass": [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["hours", "attendance", "marks"]]
y = df["pass"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved!")