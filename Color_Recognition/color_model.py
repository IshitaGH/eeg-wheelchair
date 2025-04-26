# train_svm.py
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# 1. Load your processed data
pickle_path = "/Users/adhityaram/Projects/eeg-wheelchair/Color_Recognition/color_recog/processed_data.pkl"
with open(pickle_path, "rb") as f:
    data = pickle.load(f)
X, y = data['X'], data['y']

# 2. (New) Dimensionality reduction with PCA
#    Keep enough components to explain 95% of variance
pca = PCA(n_components=0.95)

# 3. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Build a pipeline: scaling → PCA → SVM
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca',    pca),
    ('svc',    SVC(kernel='linear'))
])

# 5. (New) Grid‐search over C
param_grid = {
    'svc__C': [0.01, 0.1, 1, 10, 100]
}
grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,              # 5‐fold cross‐validation on the training set
    scoring='accuracy',
    n_jobs=-1          # parallelize across cores
)

# 6. Fit on TRAINING data only
grid.fit(X_train, y_train)

print("Best C:", grid.best_params_['svc__C'])
print("CV accuracy (on train folds):", grid.best_score_)

# 7. Evaluate best estimator on the held-out TEST set
best_model = grid.best_estimator_
test_acc = best_model.score(X_test, y_test)
print(f"Test‐set accuracy: {test_acc:.2%}")