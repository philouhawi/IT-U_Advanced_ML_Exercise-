import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from pandas import DataFrame

class MLP:
    data_path: str
    df: DataFrame
    preprocessor: ColumnTransformer
    train_loader: DataLoader
    # training dataset
    x_train: DataFrame
    y_train: DataFrame
    # validation dataset
    x_val: DataFrame
    y_val: DataFrame
    # test dataset
    x_test: DataFrame
    y_test: DataFrame

    x_val_tensor: torch.Tensor
    y_val_tensor: torch.Tensor

    torch.manual_seed(42)
    np.random.seed(42)

    def __init__(self, data_path):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)

    def inspect_dataset(self):

        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.expand_frame_repr', False)
        #inspecting my data
        print("Dataframe head")
        print(self.df.head(10))

        # Droping the useless column 'Unnamed: 0'
        self.df = self.df.drop('Unnamed: 0', axis=1)

        print("\nDataframe Info")
        self.df.info()

        print("\nDataframe Description")
        print(self.df.describe())

        print("\nshape")
        self.df.shape

        print("\nData types:")
        print(self.df.dtypes)

    def plot_data(self, target):
        print(self.df[target].describe())
        x_label = "duration"
        y_label = "price"
        plt.figure(figsize=(8, 4))
        # histogram: showing the distribution of one variable
        plt.hist(self.df[target].dropna(), bins=40)
        # plt.scatter(
        #     self.df[x_label],
        #     self.df[y_label],
        # )
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Distribution of flight fares")
        plt.tight_layout()
        plt.show()

    def split_data(self):
        features = self.df.drop(columns=["price"]).copy()
        target = self.df["price"].copy()

        # 15% test set
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.15, random_state=42)

        # 15% of remaining into validation set
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15/0.85, random_state=42)

        print("Training:", x_train.shape)
        print("Validation:", x_val.shape)
        print("Test:", x_test.shape)

        return [x_train, y_train], [x_val, y_val], [x_test, y_test]

    def data_preprocessing(self):
        # convert target data: price to numeric value
        self.df["price"] = pd.to_numeric(self.df["price"], errors="coerce")

        # delete row where the price is missing
        self.df = self.df.dropna(subset=["price"]).copy()

        # delete duplicate row
        self.df = self.df.drop_duplicates().copy()

        self._check_feature_value_uniqness(self.df)

        # drop flight
        self.df = self.df.drop(columns=["flight"])

    def flight_fair_mlp_model(self):
        # Fit the preprocessor using training data only.
        X_train_processed = self.preprocessor.fit_transform(self.x_train)
        X_val_processed = self.preprocessor.transform(self.x_val)

        # Convert to dense arrays if memory permits.
        if hasattr(X_train_processed, "toarray"):
            X_train_processed = X_train_processed.toarray()

        if hasattr(X_val_processed, "toarray"):
            X_val_processed = X_val_processed.toarray()

        X_train_np = np.asarray(X_train_processed, dtype=np.float32)
        X_val_np = np.asarray(X_val_processed, dtype=np.float32)

        y_train_np = self.y_train.to_numpy(dtype=np.float32).reshape(-1, 1)
        y_val_np = self.y_val.to_numpy(dtype=np.float32).reshape(-1, 1)

        X_train_tensor = torch.from_numpy(X_train_np)
        y_train_tensor = torch.from_numpy(y_train_np)

        self.x_val_tensor = torch.from_numpy(X_val_np)
        self.y_val_tensor = torch.from_numpy(y_val_np)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True
        )

        model = FlightFareMLP(
            input_dim=X_train_tensor.shape[1]
        )

        total_params = sum(
            p.numel() for p in model.parameters()
            if p.requires_grad
        )

        print("Input features:", X_train_tensor.shape[1])
        print("Trainable parameters:", total_params)
        print(model)
        return model

    def train_mlp(self, model):
        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.005
        )

        epochs = 100

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            for xb, yb in self.train_loader:
                optimizer.zero_grad()

                predictions = model(xb)
                loss = criterion(predictions, yb)

                loss.backward()
                optimizer.step()

                train_loss_sum += loss.item() * len(xb)
                train_count += len(xb)

            epoch_train_loss = train_loss_sum / train_count

            # Validation phase
            model.eval()

            with torch.no_grad():
                val_predictions = model(self.x_val_tensor)
                val_loss = criterion(
                    val_predictions,
                    self.y_val_tensor
                ).item()

            train_losses.append(epoch_train_loss)
            val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch + 1:3d} | "
                    f"Train MSE: {epoch_train_loss:.4f} | "
                    f"Validation MSE: {val_loss:.4f}"
                )

        # Training finished: generate validation predictions
        model.eval()

        with torch.no_grad():
            val_predictions = model(self.x_val_tensor)

        # Convert PyTorch Tensor to NumPy array
        val_pred = val_predictions.detach().cpu().numpy().flatten()
        y_val = self.y_val_tensor.detach().cpu().numpy().flatten()

        self.plot_diagnostics(y_val, val_pred, "MLP — Validation Set - LR: 0.005 - Epoch: 100")
        plt.figure(figsize=(8, 4))
        plt.plot(train_losses, label="Training MSE")
        plt.plot(val_losses, label="Validation MSE")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.title("MLP Learning Curves")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def _check_feature_value_uniqness(self, data: DataFrame):
        categorical_cols = [
            "airline",
            "flight",
            "source_city",
            "departure_time",
            "stops",
            "arrival_time",
            "destination_city",
            "class"
        ]

        for col in categorical_cols:
            print(f"{col}: {data[col].nunique()} unique values")

    def linear_regression_baseline(self):
        numeric_cols = self.x_train.select_dtypes(include="number").columns.tolist()

        categorical_cols = [
            col for col in self.x_train.columns
            if col not in numeric_cols
        ]

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols)
        ])

        baseline = Pipeline([
            ("preprocessor", self.preprocessor),
            ("model", LinearRegression())
        ])

        baseline.fit(self.x_train, self. y_train)

        val_pred = baseline.predict(self.x_val)

        mae = mean_absolute_error(self.y_val, val_pred)
        rmse = np.sqrt(mean_squared_error(self.y_val, val_pred))

        print(f"Validation MAE:  {mae:.2f}")
        print(f"Validation RMSE: {rmse:.2f}")

        self.plot_diagnostics(self.y_test, val_pred, "Baseline (Linear Regression)")

    def plot_diagnostics(self, y_true, y_pred, title):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        resid = y_true - y_pred
        m = self._regression_metrics(y_true, y_pred)

        fig, axes = plt.subplots(2, 2, figsize=(13, 10))

        # (1) 预测 vs 真实
        ax = axes[0, 0]
        ax.scatter(y_true, y_pred, s=6, alpha=0.18, color="#4C72B0", edgecolors="none")
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="perfect prediction")
        ax.set_xlabel("Actual price (INR)")
        ax.set_ylabel("Predicted price (INR)")
        ax.set_title(f"Predicted vs Actual   (R² = {m['R2']:.3f})")
        ax.legend()

        # (2) 残差 vs 预测值
        ax = axes[0, 1]
        ax.scatter(y_pred, resid, s=6, alpha=0.18, color="#55A868", edgecolors="none")
        ax.axhline(0, color="red", ls="--", lw=1.5)
        ax.set_xlabel("Predicted price (INR)")
        ax.set_ylabel("Residual  (y - ŷ)")
        ax.set_title("Residual plot（理想情况：随机散布在 0 附近）")

        # (3) 残差分布
        ax = axes[1, 0]
        sns.histplot(resid, bins=80, kde=True, ax=ax, color="#C44E52")
        ax.axvline(0, color="black", ls="--", lw=1.2)
        ax.set_xlabel("Residual")
        ax.set_title(f"Residual distribution  (mean={resid.mean():.1f}, std={resid.std():.1f})")

        # (4) 误差分位数曲线
        ax = axes[1, 1]
        abs_err = np.sort(np.abs(resid))
        pct = np.arange(1, len(abs_err) + 1) / len(abs_err) * 100
        ax.plot(abs_err, pct, lw=2, color="#8172B3")
        for q in [50, 75, 90, 95]:
            v = np.percentile(np.abs(resid), q)
            ax.axhline(q, color="gray", ls=":", lw=0.8)
            ax.annotate(f"P{q} = {v:,.0f}", xy=(v, q), xytext=(v, q - 6), fontsize=9)
        ax.set_xlabel("Absolute error (INR)")
        ax.set_ylabel("Cumulative % of samples")
        ax.set_title("误差累积分布（看长尾有多重）")

        fig.suptitle(
            f"{title}\nMAE = {m['MAE']:,.0f} | RMSE = {m['RMSE']:,.0f} | "
            f"R² = {m['R2']:.3f} | MAPE = {m['MAPE']:.1f}%",
            fontsize=13, y=1.00
        )
        plt.tight_layout()
        plt.show()
        return m

        metrics_baseline = plot_diagnostics(y_val, val_pred, "Baseline — Linear Regression")

    def _regression_metrics(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return {
            "MAE": mean_absolute_error(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "R2": r2_score(y_true, y_pred),
            "MAPE": float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-9, None))) * 100),
        }

    def _convert_duration_to_minute(self, value):
        if pd.isna(value):
            return np.nan

        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value) * 60

class FlightFareMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)

