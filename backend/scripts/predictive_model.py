# src/predictive_model.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


class ETaPredictor:
    """Train and use a random-forest model to predict ETa from project ETa/NDVI/weather data."""

    def __init__(self):
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
        )
        self.feature_names = None
        self.target_name = "eta_actual"
        self.ndvi_median_ = None

    def _coerce_datetime(self, df):
        """Convert date-like columns to pandas datetime values."""
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "date_dt" in df.columns:
            df["date"] = pd.to_datetime(df["date_dt"], errors="coerce")
        return df

    def _get_target_column(self, df):
        """Return the first supported ETa target column found in the input data."""
        for name in ("eta_actual", "eta", "eta_mm"):
            if name in df.columns:
                return name
        raise ValueError("No target column found. Expected one of: eta_actual, eta, eta_mm")

    def _build_feature_frame(self, historical_data, weather_data=None, include_target=True):
        """Build a feature matrix compatible with the project ETa/NDVI/weather dataset."""
        df = self._coerce_datetime(historical_data.copy())

        if weather_data is not None:
            weather_df = self._coerce_datetime(weather_data.copy())
            if "date" in weather_df.columns:
                df = df.merge(weather_df, on="date", how="left")
            else:
                raise ValueError("weather_data must contain a date column")

        df = df.dropna(subset=["date"]).reset_index(drop=True)

        if "date" not in df.columns:
            raise ValueError("Input data must contain a date column")

        target_col = self._get_target_column(df)

        feature_frame = pd.DataFrame(index=df.index)

        if "ndvi" in df.columns:
            feature_frame["ndvi"] = pd.to_numeric(df["ndvi"], errors="coerce")
        elif "NDVI" in df.columns:
            feature_frame["ndvi"] = pd.to_numeric(df["NDVI"], errors="coerce")
        else:
            feature_frame["ndvi"] = np.nan

        if "eto" in df.columns:
            feature_frame["eto"] = pd.to_numeric(df["eto"], errors="coerce")
        elif "temperature_2m_mean" in df.columns:
            feature_frame["eto"] = pd.to_numeric(df["temperature_2m_mean"], errors="coerce")
        else:
            feature_frame["eto"] = np.nan

        if "rain_sum" in df.columns:
            feature_frame["rain_sum"] = pd.to_numeric(df["rain_sum"], errors="coerce")
        else:
            feature_frame["rain_sum"] = np.nan

        feature_frame["day_of_year"] = df["date"].dt.dayofyear
        feature_frame["month"] = df["date"].dt.month
        feature_frame["year"] = df["date"].dt.year

        for col in ("season", "cultivar", "block_id"):
            if col in df.columns:
                feature_frame = pd.concat(
                    [feature_frame, pd.get_dummies(df[col].astype(str), prefix=col)],
                    axis=1,
                )

        numeric_cols = feature_frame.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            medians = feature_frame[numeric_cols].median()
            feature_frame[numeric_cols] = feature_frame[numeric_cols].fillna(medians)

        if include_target:
            feature_frame[self.target_name] = pd.to_numeric(df[target_col], errors="coerce")
            feature_frame = feature_frame.dropna(subset=[self.target_name]).reset_index(drop=True)
            self.ndvi_median_ = feature_frame["ndvi"].median()
            return feature_frame

        return feature_frame

    def prepare_training_data(self, historical_data, weather_data=None):
        """
        Prepare training data from the ETa/NDVI dataset and optional weather data.

        This method is compatible with the project CSVs in data/historic_eta_data.csv
        and data/historic_weather_data.csv.
        """
        return self._build_feature_frame(
            historical_data,
            weather_data=weather_data,
            include_target=True,
        )

    def train(self, historical_data, weather_data=None):
        """Train the ETa prediction model using the project dataset and report metrics."""
        df = self.prepare_training_data(historical_data, weather_data=weather_data)

        feature_cols = [col for col in df.columns if col != self.target_name]
        X = df[feature_cols].values
        y = df[self.target_name].values

        self.feature_names = feature_cols

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("✅ ETa Model Trained")
        print(f"   MAE: {mae:.3f} mm/day")
        print(f"   R²: {r2:.3f}")
        print(f"   Features: {len(feature_cols)}")

        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)

        print("\nTop Features:")
        print(importance.head(5))

        return self.model, {"mae": mae, "r2": r2}

    def predict_eta(self, features_df):
        """
        Predict ETa for a DataFrame of feature rows.

        The DataFrame can either contain the same feature columns used during training
        or the original project-style columns, which will be converted automatically.
        """
        if self.feature_names is None:
            raise ValueError("The model must be trained before calling predict_eta")

        if isinstance(features_df, pd.DataFrame):
            if set(self.feature_names).issubset(features_df.columns):
                X = features_df[self.feature_names]
            else:
                prepared_features = self._build_feature_frame(features_df, include_target=False)
                X = prepared_features[self.feature_names]
        else:
            raise TypeError("features_df must be a pandas DataFrame")

        predictions = self.model.predict(X.values)
        return predictions

    def predict_future_eta(self, future_data, weather_data=None):
        """
        Predict ETa for future rows using the same schema as the training data.

        If future_data is a DataFrame with project-style columns, it will be prepared
        automatically and the predictions will be returned with the feature frame.
        """
        if isinstance(future_data, pd.DataFrame):
            prepared = self._build_feature_frame(future_data, weather_data=weather_data, include_target=False)
            prepared["predicted_eta"] = self.predict_eta(prepared)
            return prepared

        raise TypeError("future_data must be a pandas DataFrame")