import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss


df = pd.read_csv(__import__('pathlib').Path(__file__).resolve().parents[2] / 'AI_datasets/RouterBench/routerbench_0shot.csv')

models = [c for c in df.columns if c not in
          ["sample_id", "prompt", "eval_name",
           "oracle_model_to_route_to"] and "|" not in c]

df = df.melt(
    id_vars=["sample_id", "prompt", "eval_name"],
    value_vars=models,
    var_name="model_id",
    value_name="score"
)

df["quality_label"] = (df["score"] == 1).astype(int)
df["token_length"] = df["prompt"].str.len()


ids = df["sample_id"].unique()
train_ids, test_ids = train_test_split(
    ids, test_size=0.2, random_state=42
)

train = df[df.sample_id.isin(train_ids)].copy()
test = df[df.sample_id.isin(test_ids)].copy()


model_codes, model_names = pd.factorize(train["model_id"])
item_codes, item_names = pd.factorize(train["sample_id"])

y = train["quality_label"].to_numpy()
M, N = len(model_names), len(item_names)

def objective(params):
    theta = params[:M]
    b = params[M:]
    logits = theta[model_codes] - b[item_codes]
    p = expit(logits)

    loss = np.logaddexp(0, logits).sum() - y @ logits
    loss += 0.1 * np.sum(params ** 2)

    error = p - y
    grad_theta = np.bincount(
        model_codes, weights=error, minlength=M
    )
    grad_b = -np.bincount(
        item_codes, weights=error, minlength=N
    )
    grad = np.r_[grad_theta, grad_b] + 0.2 * params
    return loss, grad

result = minimize(
    objective, np.zeros(M + N),
    jac=True, method="L-BFGS-B"
)

theta = dict(zip(model_names, result.x[:M]))
difficulty = dict(zip(item_names, result.x[M:]))


items = train.drop_duplicates("sample_id").copy()
items["difficulty"] = items["sample_id"].map(difficulty)

features = ["token_length", "eval_name"]

items["eval_name"] = items["eval_name"].astype("category")
test["eval_name"] = pd.Categorical(
    test["eval_name"],
    categories=items["eval_name"].cat.categories
)

gbm = LGBMRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    verbosity=-1
)

gbm.fit(items[features], items["difficulty"])


b_pred = gbm.predict(test[features])
theta_pred = test["model_id"].map(theta).to_numpy()

pred = expit(theta_pred - b_pred)
y_test = test["quality_label"]

print("AUC:", roc_auc_score(y_test, pred))
print("Log Loss:", log_loss(y_test, pred))
print("Brier Score:", brier_score_loss(y_test, pred))
