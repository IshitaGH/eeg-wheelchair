import pickle
import pandas as pd

# 1. Load the pickled data (update the path below)
pickle_path = "/Users/adhityaram/Projects/eeg-wheelchair/Color Recognition /color_recog/processed_data.pkl"
with open(pickle_path, "rb") as f:
    data = pickle.load(f)

# 2. Extract feature matrix X and label vector y
X = data["X"]  # shape (n_epochs, n_features)
y = data["y"]  # shape (n_epochs,)

# 3. Create a DataFrame
df = pd.DataFrame(X)   # columns 0…(n_features-1)
df["label"] = y        # adds a 'label' column

# 4. Preview in console
print(df.head())