import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from data_utils import augment_data_with_noise
from model import MLP


# ======================================================
# 1) LOAD & CLEAN DATA
# ======================================================
data = pd.read_csv("data.csv", encoding="latin1")

print("🧾 Columns before renaming:", list(data.columns))

data.rename(columns={
    "Current Density (mA/cm²)": "J",
    "Voltage (V)": "V",
    "Porosity (%)": "Porosity",
    "Pore Size (µm)": "PoreSize"
}, inplace=True)

print("✅ Columns after renaming:", list(data.columns))

input_cols = ["J", "V"]
target_cols = ["Porosity", "PoreSize"]


# ======================================================
# 2) METRICS
# ======================================================
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


# ======================================================
# 3) TRAIN + EVALUATE ONE CONFIGURATION USING LOOCV
# ======================================================
def train_and_evaluate_loocv(data, input_cols, target_cols,
                             aug_factor, hidden_size, dropout_rate,
                             lr=1e-3, epochs=5000):
    preds_list = []

    for i in range(len(data)):
        test_df = data.iloc[[i]]
        train_df = data.drop(i)

        # ------------------------------------------
        # Augment training data only
        # ------------------------------------------
        if aug_factor > 0:
            aug_train = augment_data_with_noise(
                train_df,
                input_cols,
                augmentation_factor=aug_factor
            )
            train_df = pd.concat([train_df, aug_train], ignore_index=True)

        # ------------------------------------------
        # Fit scalers on training fold only
        # ------------------------------------------
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_x.fit_transform(train_df[input_cols])
        y_train = scaler_y.fit_transform(train_df[target_cols])

        X_test = scaler_x.transform(test_df[input_cols])

        # ------------------------------------------
        # Convert to tensors
        # ------------------------------------------
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)

        # ------------------------------------------
        # Build model
        # ------------------------------------------
        model = MLP(
            input_size=len(input_cols),
            output_size=len(target_cols),
            hidden_neurons=hidden_size,
            dropout_rate=dropout_rate
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        # ------------------------------------------
        # Training loop
        # ------------------------------------------
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            output = model(X_train_t)
            loss = criterion(output, y_train_t)
            loss.backward()
            optimizer.step()

        # ------------------------------------------
        # Predict on left-out sample
        # ------------------------------------------
        model.eval()
        with torch.no_grad():
            pred_scaled = model(X_test_t).cpu().numpy()

        pred_original = scaler_y.inverse_transform(pred_scaled)
        preds_list.append(pred_original.flatten())

    return np.array(preds_list)


# ======================================================
# 4) ABLATION CONFIGURATIONS
# ======================================================
configs = [
    {
        "name": "Optimized model (with dropout)",
        "aug": 30,
        "neurons": 16,
        "dropout": 0.2
    },
    {
        "name": "Optimized model (without augmentation)",
        "aug": 0,
        "neurons": 16,
        "dropout": 0.2
    },
    {
        "name": "Smaller network",
        "aug": 30,
        "neurons": 8,
        "dropout": 0.2
    },
    {
        "name": "Optimized model (without dropout)",
        "aug": 30,
        "neurons": 16,
        "dropout": 0.0
    }
]


# ======================================================
# 5) RUN ALL EXPERIMENTS
# ======================================================
results_rows = []

y_true = data[target_cols].values
y_true_porosity = y_true[:, 0]
y_true_poresize = y_true[:, 1]

for cfg in configs:
    print(f"\n🚀 Running config: {cfg['name']}")

    preds = train_and_evaluate_loocv(
        data=data,
        input_cols=input_cols,
        target_cols=target_cols,
        aug_factor=cfg["aug"],
        hidden_size=cfg["neurons"],
        dropout_rate=cfg["dropout"],
        lr=1e-3,
        epochs=5000
    )

    pred_porosity = preds[:, 0]
    pred_poresize = preds[:, 1]

    # Per-output metrics
    mae_por = mae(y_true_porosity, pred_porosity)
    rmse_por = rmse(y_true_porosity, pred_porosity)
    r2_por = r2_score(y_true_porosity, pred_porosity)

    mae_ps = mae(y_true_poresize, pred_poresize)
    rmse_ps = rmse(y_true_poresize, pred_poresize)
    r2_ps = r2_score(y_true_poresize, pred_poresize)

    r2_avg = np.nanmean([r2_por, r2_ps])

    results_rows.append({
        "Model configuration": cfg["name"],
        "R² (Porosity)": r2_por,
        "MAE (Porosity)": mae_por,
        "RMSE (Porosity)": rmse_por,
        "R² (Pore size)": r2_ps,
        "MAE (Pore size)": mae_ps,
        "RMSE (Pore size)": rmse_ps,
        "Average R²": r2_avg
    })


# ======================================================
# 6) SAVE FINAL TABLE
# ======================================================
table4 = pd.DataFrame(results_rows)
table4.to_csv("ablation_results_table4_optimized.csv", index=False)

print("\n✅ Optimized Table 4 saved → ablation_results_table4_optimized.csv")
print(table4)

