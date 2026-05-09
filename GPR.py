import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ==========================
# Load dataset
# ==========================
df = pd.read_csv("data.csv", encoding="latin1")

df.rename(columns={
    "Current Density (mA/cm²)": "J",
    "Voltage (V)": "V",
    "Porosity (%)": "Porosity",
    "Pore Size (µm)": "PoreSize"
}, inplace=True)

X = df[['J', 'V']].values
y_por = df['Porosity'].values
y_pore = df['PoreSize'].values


# ==========================
# GPR evaluation with LOOCV
# ==========================
def evaluate_gpr(X, y):

    loo = LeaveOneOut()

    preds = []
    truths = []

    for train_idx, test_idx in loo.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # scale inputs
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)

        gpr = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=10
        )

        gpr.fit(X_train, y_train)

        pred = gpr.predict(X_test)[0]

        preds.append(pred)
        truths.append(y_test[0])

    preds = np.array(preds)
    truths = np.array(truths)

    return {
        "MAE": mean_absolute_error(truths, preds),
        "RMSE": np.sqrt(mean_squared_error(truths, preds)),
        "R2": r2_score(truths, preds)
    }


# ==========================
# Run evaluation
# ==========================
results_por = evaluate_gpr(X, y_por)
results_pore = evaluate_gpr(X, y_pore)

print("\nGPR Porosity:", results_por)
print("GPR Pore Size:", results_pore)

