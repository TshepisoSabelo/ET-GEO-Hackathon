# src/optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class IrrigationOptimizer:
    """Optimize irrigation schedules using ETa, rainfall, and soil-moisture constraints."""

    def __init__(
        self,
        block,
        eta_predictions,
        horizon_days=7,
        field_capacity=None,
        wilting_point=None,
        max_irrigation=25.0,
        current_date=None,
        stage_lookup=None,
    ):
        """
        Args:
            block: Block-like object or dictionary with soil properties and stage information.
            eta_predictions: Predicted ETa values for each day, as a sequence, Series, or DataFrame.
            horizon_days: Planning horizon in days.
            field_capacity: Optional field-capacity value in mm.
            wilting_point: Optional wilting-point value in mm.
            max_irrigation: Maximum irrigation amount allowed per day in mm.
            current_date: Optional starting date for the forecast horizon.
            stage_lookup: Optional callable that returns a growth stage for a given date.
        """
        self.block = block
        self.horizon = max(1, int(horizon_days))
        self.eta_pred = self._coerce_numeric_series(eta_predictions, self.horizon)
        self.max_irrigation = float(max_irrigation)

        self.field_capacity = self._resolve_numeric_value(
            field_capacity, getattr(block, "field_capacity", None), 100.0
        )
        self.wilting_point = self._resolve_numeric_value(
            wilting_point, getattr(block, "wilting_point", None), 40.0
        )

        self.current_date = self._resolve_current_date(current_date, block)
        self.stage_lookup = stage_lookup or getattr(block, "get_stage", None)
        self.stage_bounds = self.get_stage_bounds()

    def _resolve_current_date(self, current_date, block):
        """Resolve a usable current date from the provided block or argument."""
        if current_date is not None:
            return pd.Timestamp(current_date)

        if block is None:
            return pd.Timestamp.today().normalize()

        if hasattr(block, "current_date") and getattr(block, "current_date") is not None:
            return pd.Timestamp(getattr(block, "current_date"))

        if isinstance(block, dict) and block.get("current_date") is not None:
            return pd.Timestamp(block["current_date"])

        return pd.Timestamp.today().normalize()

    def _resolve_numeric_value(self, explicit_value, fallback_value, default_value):
        """Resolve a numeric soil property from explicit values or object attributes."""
        if explicit_value is not None:
            return float(explicit_value)
        if fallback_value is not None:
            return float(fallback_value)
        return float(default_value)

    def _coerce_numeric_series(self, values, horizon):
        """Convert ETa/rainfall inputs into a numeric array of the requested horizon length."""
        if values is None:
            return np.zeros(horizon, dtype=float)

        if isinstance(values, pd.DataFrame):
            if "eta_mm" in values.columns:
                values = values["eta_mm"]
            elif "predicted_eta" in values.columns:
                values = values["predicted_eta"]
            elif "eta" in values.columns:
                values = values["eta"]
            else:
                values = values.iloc[:, 0]
        elif isinstance(values, pd.Series):
            values = values
        else:
            values = np.asarray(values, dtype=float).reshape(-1)

        if isinstance(values, pd.Series):
            series = values.astype(float).to_numpy()
        else:
            series = np.asarray(values, dtype=float).reshape(-1)

        if len(series) < horizon:
            series = np.pad(series, (0, horizon - len(series)), constant_values=series[-1] if len(series) > 0 else 0.0)
        elif len(series) > horizon:
            series = series[:horizon]

        return series.astype(float)

    def _resolve_stage(self, day_offset=0):
        """Resolve the growth stage for the requested day offset."""
        date = self.current_date + pd.Timedelta(days=day_offset)

        if callable(self.stage_lookup):
            return str(self.stage_lookup(date)).lower()

        if isinstance(self.block, dict):
            stage = self.block.get("stage") or self.block.get("growth_stage")
            if stage is not None:
                return str(stage).lower()

        if hasattr(self.block, "get_stage") and callable(getattr(self.block, "get_stage")):
            return str(self.block.get_stage(date)).lower()

        if hasattr(self.block, "stage"):
            return str(getattr(self.block, "stage")).lower()

        return "dormant"

    def get_stage_bounds(self, stage=None):
        """Get soil-moisture bounds for a given stage as fractions of field capacity."""
        lower_bounds = {
            "dormant": 0.25,
            "budbreak": 0.40,
            "flowering": 0.50,
            "pre-veraison": 0.35,
            "veraison": 0.40,
            "harvest": 0.30,
            "leafdrop": 0.25,
        }

        upper_bounds = {
            "dormant": 0.70,
            "budbreak": 0.80,
            "flowering": 0.85,
            "pre-veraison": 0.75,
            "veraison": 0.80,
            "harvest": 0.70,
            "leafdrop": 0.65,
        }

        if stage is None:
            stage = "dormant"

        stage_key = str(stage).lower().replace(" ", "")
        if stage_key not in lower_bounds:
            stage_key = "dormant"

        lower = lower_bounds[stage_key]
        upper = upper_bounds[stage_key]
        return lower, upper

    def get_soil_bounds(self, stage):
        """Return lower and upper allowable soil-moisture bounds for a stage."""
        lower, upper = self.get_stage_bounds(stage)
        return self.field_capacity * lower, self.field_capacity * upper

    def _simulate_soil_moisture(self, irrigation_schedule, initial_soil_moisture, rainfall_forecast):
        """Simulate soil-moisture evolution across the planning horizon."""
        moisture = float(initial_soil_moisture)
        path = []

        for day in range(self.horizon):
            stage = self._resolve_stage(day)
            lower, upper = self.get_soil_bounds(stage)

            moisture -= self.eta_pred[day]
            moisture += float(irrigation_schedule[day])
            moisture += float(rainfall_forecast[day])
            moisture = min(moisture, self.field_capacity)
            path.append(moisture)

            if moisture < lower:
                return path, True
            if moisture > upper:
                return path, False

        return path, False

    def optimize(self, initial_soil_moisture, rainfall_forecast):
        """
        Find an irrigation schedule that keeps soil moisture within the target bounds.

        Returns: a list of irrigation amounts for each day in the horizon.
        """
        rainfall = self._coerce_numeric_series(rainfall_forecast, self.horizon)

        def objective(schedule):
            total_water = np.sum(schedule)
            penalty = 0.0
            moisture = float(initial_soil_moisture)

            for day in range(self.horizon):
                stage = self._resolve_stage(day)
                lower, upper = self.get_soil_bounds(stage)

                moisture -= self.eta_pred[day]
                moisture += float(schedule[day])
                moisture += float(rainfall[day])

                if moisture < 0:
                    penalty += 1000 * abs(moisture)
                if moisture < lower:
                    penalty += (lower - moisture) ** 2 * 100
                if moisture > upper:
                    penalty += (moisture - upper) ** 2 * 100

            return total_water + penalty

        initial_guess = np.zeros(self.horizon, dtype=float)
        bounds = [(0.0, self.max_irrigation) for _ in range(self.horizon)]

        result = minimize(
            objective,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
        )

        if result.success:
            return result.x.tolist()
        return self.heuristic_optimize(initial_soil_moisture, rainfall)

    def heuristic_optimize(self, initial_soil_moisture, rainfall_forecast):
        """Fallback heuristic optimizer that always returns a feasible-looking schedule."""
        rainfall = self._coerce_numeric_series(rainfall_forecast, self.horizon)
        schedule = []
        soil_moisture = float(initial_soil_moisture)

        for day in range(self.horizon):
            stage = self._resolve_stage(day)
            lower, upper = self.get_soil_bounds(stage)

            soil_moisture -= self.eta_pred[day]
            soil_moisture += float(rainfall[day])

            if soil_moisture < lower:
                irrigation = min(self.max_irrigation, max(0.0, (lower + upper) / 2 - soil_moisture))
                schedule.append(irrigation)
                soil_moisture += irrigation
            else:
                schedule.append(0.0)

        return schedule

    def optimize_with_ndvi_penalty(self, initial_soil_moisture, rainfall_forecast, ndvi_target=0.6):
        """
        Optimize irrigation with an additional penalty for NDVI losses below a target.

        This keeps the schedule irrigation-efficient while still protecting canopy health.
        """
        rainfall = self._coerce_numeric_series(rainfall_forecast, self.horizon)

        def objective(schedule):
            water_cost = np.sum(schedule)
            soil_moisture = float(initial_soil_moisture)
            ndvi = float(ndvi_target)

            for day in range(self.horizon):
                stage = self._resolve_stage(day)
                lower, upper = self.get_soil_bounds(stage)

                soil_moisture -= self.eta_pred[day]
                soil_moisture += float(schedule[day])
                soil_moisture += float(rainfall[day])

                stress_factor = max(0.0, 1 - (soil_moisture / self.field_capacity))
                ndvi += self.ndvi_response_model(stress_factor, stage)

                if ndvi < ndvi_target:
                    water_cost += (ndvi_target - ndvi) * 100
                if soil_moisture < lower:
                    water_cost += (lower - soil_moisture) ** 2 * 50
                if soil_moisture > upper:
                    water_cost += (soil_moisture - upper) ** 2 * 50

            return water_cost

        initial_guess = np.zeros(self.horizon, dtype=float)
        bounds = [(0.0, self.max_irrigation) for _ in range(self.horizon)]

        result = minimize(
            objective,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
        )

        if result.success:
            return result.x.tolist()
        return self.heuristic_optimize(initial_soil_moisture, rainfall)

    def ndvi_response_model(self, stress_factor, stage):
        """Approximate NDVI response to stress for the current growth stage."""
        stage_weight = {
            "dormant": 0.5,
            "budbreak": 0.8,
            "flowering": 1.0,
            "pre-veraison": 0.9,
            "veraison": 0.8,
            "harvest": 0.6,
            "leafdrop": 0.5,
        }.get(str(stage).lower().replace(" ", ""), 0.7)
        return -0.02 * stress_factor * stage_weight