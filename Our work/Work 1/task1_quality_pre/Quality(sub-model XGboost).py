import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv(__import__('pathlib').Path(__file__).resolve().parents[2] / 'AI_datasets/RouterBench/routerbench_0shot.csv')

exclude = [
    "sample_id",
    "prompt",
    "eval_name",
    "oracle_model_to_route_to"
]

models = [
    c for c in df.columns
    if c not in exclude and "|" not in c
]

df = df.melt(
    id_vars=["sample_id", "prompt", "eval_name"],
    value_vars=models,
    var_name="model_id",
    value_name="score"
)

df["score"] = pd.to_numeric(df["score"])
df["token_length"] = df["prompt"].fillna("").str.len()


ids = df["sample_id"].unique()

train_ids, test_ids = train_test_split(
    ids,
    test_size=0.2,
    random_state=42
)

train = df[df["sample_id"].isin(train_ids)].copy()
test = df[df["sample_id"].isin(test_ids)].copy()


features = [
    "token_length",
    "eval_name",
    "model_id"
]


train_lgb = train[features].copy()
test_lgb = test[features].copy()

for col in ["eval_name", "model_id"]:
    categories = pd.Categorical(df[col]).categories
    train_lgb[col] = pd.Categorical(
        train_lgb[col], categories=categories
    )
    test_lgb[col] = pd.Categorical(
        test_lgb[col], categories=categories
    )



encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

train_xgb = train[features].copy()
test_xgb = test[features].copy()

cat_cols = ["eval_name", "model_id"]

train_xgb[cat_cols] = encoder.fit_transform(
    train_xgb[cat_cols]
)

test_xgb[cat_cols] = encoder.transform(
    test_xgb[cat_cols]
)


for col in cat_cols:
    train_xgb[col] = train_xgb[col].astype("int32")
    test_xgb[col] = test_xgb[col].astype("int32")


lgb_model = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    verbosity=-1
)

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    tree_method="hist",
    enable_categorical=True,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

print("Training LightGBM...")
lgb_model.fit(train_lgb, train["score"])

print("Training XGBoost...")
xgb_model.fit(train_xgb, train["score"])


pred_lgb = np.clip(
    lgb_model.predict(test_lgb), 0, 1
)

pred_xgb = np.clip(
    xgb_model.predict(test_xgb), 0, 1
)


y_true = test["score"].to_numpy()

def evaluate(name, pred):
    mae = mean_absolute_error(y_true, pred)
    rmse = np.sqrt(mean_squared_error(y_true, pred))

    print(f"\n{name}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")

    return {
        "Model": name,
        "MAE": mae,
        "RMSE": rmse
    }

results = [
    evaluate("LightGBM", pred_lgb),
    evaluate("XGBoost", pred_xgb)
]

result_df = pd.DataFrame(results)

print("\n========== Model Comparison ==========")
print(result_df.to_string(index=False))


result_df.to_csv(
    "quality_xgboost_comparison.csv",
    index=False
)

output = test[
    ["sample_id", "model_id", "eval_name", "score"]
].copy()

output["LightGBM_prediction"] = pred_lgb
output["XGBoost_prediction"] = pred_xgb

output.to_csv(
    "quality_xgboost_predictions.csv",
    index=False
)

print("\nResults saved.")
