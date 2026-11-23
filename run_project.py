"""Tempering hardness prediction: LR vs SVR vs Random Forest vs GPR."""
import warnings, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "Tempering_data_for_carbon_and_low_alloy_steels.xlsx"
OUT = ROOT / "results"; OUT.mkdir(exist_ok=True)
SEED = 42

# 1. Load & clean
df = pd.read_excel(DATA)
df.columns = [c.replace("Âº", "°") for c in df.columns]
target = "Final hardness (HRC) - post tempering"
drop = ["Source", "Steel type", "Initial hardness (HRC) - post quenching"]  # ids / mostly '?'
df = df.drop(columns=drop).dropna()
print("Records:", len(df))
df["log_time"] = np.log10(df["Tempering time (s)"])   # time spans 600 s .. 115200 s
features = [c for c in df.columns if c not in (target, "Tempering time (s)")]
X, y = df[features], df[target]

# 2. Split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)

# 3. Models
models = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "SVR": make_pipeline(StandardScaler(), SVR(C=50, epsilon=0.5, gamma="scale")),
    "Random Forest": RandomForestRegressor(n_estimators=500, random_state=SEED, n_jobs=-1),
    "GPR": make_pipeline(StandardScaler(), GaussianProcessRegressor(
        kernel=C(1.0) * RBF(length_scale=np.ones(len(features))) + WhiteKernel(0.1),
        normalize_y=True, n_restarts_optimizer=1, random_state=SEED)),
}
rows, preds = [], {}
for name, m in models.items():
    m.fit(Xtr, ytr); p = m.predict(Xte); preds[name] = p
    rows.append({"Model": name, "R2": r2_score(yte, p),
                 "MAE (HRC)": mean_absolute_error(yte, p),
                 "RMSE (HRC)": np.sqrt(mean_squared_error(yte, p))})
res = pd.DataFrame(rows).round(4)
res.to_csv(OUT / "model_comparison.csv", index=False)
print(res.to_string(index=False))

# 4. Plots
best = res.sort_values("R2", ascending=False).iloc[0]["Model"]
plt.figure(figsize=(5.5, 5)); plt.scatter(yte, preds[best], s=12, alpha=.7)
lim = [yte.min(), yte.max()]; plt.plot(lim, lim, "r--")
plt.xlabel("Actual HRC"); plt.ylabel("Predicted HRC"); plt.title(f"{best}: actual vs predicted")
plt.tight_layout(); plt.savefig(OUT / "actual_vs_predicted.png", dpi=150); plt.close()

res.set_index("Model")[["R2"]].plot.bar(legend=False, figsize=(6, 4)); plt.title("R² comparison")
plt.ylabel("R²"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(OUT / "r2_comparison.png", dpi=150); plt.close()

res.set_index("Model")[["MAE (HRC)", "RMSE (HRC)"]].plot.bar(figsize=(6, 4)); plt.title("Error comparison")
plt.ylabel("HRC"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(OUT / "error_comparison.png", dpi=150); plt.close()

imp = pd.Series(models["Random Forest"].feature_importances_, index=features).sort_values()
imp.plot.barh(figsize=(6, 5)); plt.title("Random Forest feature importance")
plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=150); plt.close()
print("Best model:", best, "| outputs in", OUT)
