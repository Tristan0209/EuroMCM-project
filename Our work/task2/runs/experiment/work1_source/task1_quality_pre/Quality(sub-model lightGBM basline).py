import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv("/dataset/routerbench_0shot-v2.csv")


models = [
    c for c in df.columns
    if c not in ["sample_id", "prompt", "eval_name",
                 "oracle_model_to_route_to"]
    and "|" not in c
]

df = df.melt(
    id_vars=["sample_id", "prompt", "eval_name"],
    value_vars=models,
    var_name="model_id",
    value_name="score"
)


df["token_length"] = df["prompt"].str.len()

features = ["token_length", "eval_name", "model_id"]

df["eval_name"] = df["eval_name"].astype("category")
df["model_id"] = df["model_id"].astype("category")


ids = df["sample_id"].unique()

train_ids, test_ids = train_test_split(
    ids,
    test_size=0.2,
    random_state=42
)

train = df[df["sample_id"].isin(train_ids)]
test = df[df["sample_id"].isin(test_ids)]


model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    verbosity=-1
)

model.fit(train[features], train["score"])


pred = model.predict(test[features])
pred = np.clip(pred, 0, 1)


y_true = test["score"]

print("LightGBM Regression Baseline")
print("MAE:", mean_absolute_error(y_true, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, pred)))
