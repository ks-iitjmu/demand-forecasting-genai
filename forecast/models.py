"""Forecasting models. Each takes the feature table and a forecast origin, and returns one
row per series and forecast day with a point forecast and an 80% interval (lo, hi).

  seasonal_naive  repeat the last observed week (the baseline to beat)
  prophet         one model per series: trend, weekly and yearly seasonality, calendar
                  events as holidays, SNAP days and the price index as extra regressors
  xgboost         one global model for all series, trained to predict each day's sales
                  relative to the series' recent level, so big and small departments are
                  learned together; 10% and 90% quantile models give the interval

Every model sees only rows dated before the origin.
"""
import logging
import warnings

import numpy as np
import pandas as pd

KEYS = ["store_id", "dept_id"]
XGB_FEATURES = ["store_code", "dept_code", "wday", "month", "day_of_month", "is_weekend", "snap",
                "is_event", "event_code", "price_index", "lag_28", "lag_35", "lag_364",
                "mean_7_lag28", "mean_28_lag28", "trend_ratio", "yoy_ratio"]

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


def split(df: pd.DataFrame, origin: pd.Timestamp, horizon: int):
    """History up to and including the origin; the next `horizon` days to forecast."""
    hist = df[df["date"] <= origin]
    future = df[(df["date"] > origin) & (df["date"] <= origin + pd.Timedelta(days=horizon))]
    return hist, future


# ------------------------------------------------------------------------- baseline
def seasonal_naive(df: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    hist, future = split(df, origin, horizon)
    out = []
    for key, g in future.groupby(KEYS):
        last_week = hist[(hist.store_id == key[0]) & (hist.dept_id == key[1])].sort_values("date").tail(7)
        by_wday = dict(zip(last_week["wday"], last_week["units"]))
        # interval: the spread of the same weekday over the last 8 weeks
        recent = hist[(hist.store_id == key[0]) & (hist.dept_id == key[1])].sort_values("date").tail(56)
        lo = recent.groupby("wday")["units"].quantile(0.1).to_dict()
        hi = recent.groupby("wday")["units"].quantile(0.9).to_dict()
        out.append(g[KEYS + ["date"]].assign(yhat=g["wday"].map(by_wday).values,
                                             lo=g["wday"].map(lo).values, hi=g["wday"].map(hi).values))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- prophet
def prophet_holidays(df: pd.DataFrame) -> pd.DataFrame:
    ev = df.loc[df["event_name"] != "", ["date", "event_name"]].drop_duplicates()
    return ev.rename(columns={"date": "ds", "event_name": "holiday"}).assign(lower_window=-1, upper_window=1)


def prophet_model(df: pd.DataFrame, origin: pd.Timestamp, horizon: int) -> pd.DataFrame:
    from prophet import Prophet
    for name in ("cmdstanpy", "prophet.plot"):                # per-fit log lines and a plotting notice
        logging.getLogger(name).disabled = True
    hist, future = split(df, origin, horizon)
    holidays = prophet_holidays(df)
    out = []
    for key, g in future.groupby(KEYS):
        h = hist[(hist.store_id == key[0]) & (hist.dept_id == key[1])]
        train = pd.DataFrame({"ds": h["date"], "y": h["units"], "snap": h["snap"],
                              "price_index": h["price_index"].fillna(1.0)})
        m = Prophet(holidays=holidays, yearly_seasonality=True, weekly_seasonality=True,
                    daily_seasonality=False, seasonality_mode="multiplicative", interval_width=0.8)
        m.add_regressor("snap")
        m.add_regressor("price_index")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(train)
        fut = pd.DataFrame({"ds": g["date"], "snap": g["snap"].values,
                            "price_index": g["price_index"].fillna(1.0).values})
        p = m.predict(fut)
        out.append(g[KEYS + ["date"]].assign(yhat=p["yhat"].clip(lower=0).values,
                                             lo=p["yhat_lower"].clip(lower=0).values,
                                             hi=p["yhat_upper"].clip(lower=0).values))
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- xgboost
def add_xgb_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["store_code"] = df["store_id"].astype("category").cat.codes
    df["dept_code"] = df["dept_id"].astype("category").cat.codes
    df["event_code"] = df["event_type"].astype("category").cat.codes
    df["level"] = df["mean_28_lag28"] + 1.0                         # the series' recent level
    df["trend_ratio"] = (df["mean_7_lag28"] + 1.0) / df["level"]
    df["yoy_ratio"] = (df["lag_364"] + 1.0) / df["level"]
    return df


def xgboost_model(df: pd.DataFrame, origin: pd.Timestamp, horizon: int, seed: int = 0) -> pd.DataFrame:
    from xgboost import XGBRegressor
    df = add_xgb_columns(df)
    hist, future = split(df, origin, horizon)
    hist = hist.dropna(subset=["mean_28_lag28"])
    target = hist["units"] / hist["level"]                            # relative to the recent level
    valid = hist["date"] > origin - pd.Timedelta(days=horizon)        # last 28 days: early stopping
    params = dict(n_estimators=2000, learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8,
                  colsample_bytree=0.8, random_state=seed, early_stopping_rounds=100, tree_method="hist")
    X, Xv = hist.loc[~valid, XGB_FEATURES], hist.loc[valid, XGB_FEATURES]
    y, yv = target[~valid], target[valid]

    point = XGBRegressor(objective="reg:squarederror", **params).fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    quant = XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array([0.1, 0.9]),
                         **params).fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    Xf, level = future[XGB_FEATURES], future["level"].values
    q = quant.predict(Xf)
    return future[KEYS + ["date"]].assign(
        yhat=np.clip(point.predict(Xf) * level, 0, None),
        lo=np.clip(np.minimum(q[:, 0], q[:, 1]) * level, 0, None),
        hi=np.clip(np.maximum(q[:, 0], q[:, 1]) * level, 0, None)).reset_index(drop=True)


MODELS = {"seasonal_naive": seasonal_naive, "prophet": prophet_model, "xgboost": xgboost_model}
