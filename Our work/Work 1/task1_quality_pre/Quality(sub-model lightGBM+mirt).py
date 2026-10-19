import numpy as np
import pandas as pd
import torch
from torch import nn
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


df = pd.read_csv(__import__('pathlib').Path(__file__).resolve().parents[2] / 'AI_datasets/RouterBench/routerbench_0shot.csv')

exclude = [
    "sample_id", "prompt", "eval_name",
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


def get_dimension(name):
    name = str(name).lower()
    if any(x in name for x in ["math", "algebra", "remainder"]):
        return "Math"
    if any(x in name for x in ["mbpp", "computer", "machine-learning"]):
        return "Code"
    return "General"


def score_scheme(name):
    name = str(name).lower()
    if name.startswith("mtbench"):
        return 2
    if name in ["grade-school-math", "chinese_idioms",
                "consensus_summary"]:
        return 1
    return 0


DIMS = ["Math", "Code", "General"]

LEVELS = [
    np.array([0., 1.]),
    np.linspace(0, 1, 5),
    np.linspace(0, 1, 11)
]

df["dimension"] = df["eval_name"].apply(get_dimension)
df["scheme"] = df["eval_name"].apply(score_scheme)


df["grade"] = [
    int(np.argmin(np.abs(LEVELS[int(s)] - y)))
    for s, y in zip(df["scheme"], df["score"])
]

df["eval_name"] = df["eval_name"].astype("category")
df["model_id"] = df["model_id"].astype("category")

MODEL_NAMES = list(df["model_id"].cat.categories)
MODEL_INDEX = {name: i for i, name in enumerate(MODEL_NAMES)}


ids = df["sample_id"].unique()

train_ids, test_ids = train_test_split(
    ids, test_size=0.2, random_state=SEED
)

train = df[df["sample_id"].isin(train_ids)].copy()
test = df[df["sample_id"].isin(test_ids)].copy()


class GradedMIRT(nn.Module):
    def __init__(self, num_questions):
        super().__init__()
        self.theta = nn.Parameter(
            torch.zeros(len(MODEL_NAMES), len(DIMS))
        )
        self.b = nn.Parameter(torch.zeros(num_questions))

        self.thresholds = [
            torch.tensor([0.0]),
            torch.linspace(-1.5, 1.5, 4),
            torch.linspace(-2.5, 2.5, 10)
        ]

    def forward(self, m, q, k, scheme, grade):
        z = self.theta[m, k] - self.b[q]
        loss = z.new_tensor(0.0)

        for s in range(3):
            mask = scheme == s
            if not mask.any():
                continue

            t = self.thresholds[s]
            cum = torch.sigmoid(
                z[mask, None] - t[None, :]
            )

            probs = torch.cat([
                1 - cum[:, :1],
                cum[:, :-1] - cum[:, 1:],
                cum[:, -1:]
            ], dim=1)

            chosen = probs.gather(
                1, grade[mask, None]
            ).squeeze(1)

            loss = loss - torch.log(
                chosen.clamp_min(1e-8)
            ).sum()

        loss = loss / len(m)
        loss += 0.01 * self.theta.square().mean()
        loss += 0.003 * self.b.square().mean()
        return loss


def fit_mirt(data, epochs=20):
    question_codes, question_names = pd.factorize(
        data["sample_id"]
    )

    model_codes = data["model_id"].map(
        MODEL_INDEX
    ).to_numpy(dtype=np.int64)

    dim_codes = pd.Categorical(
        data["dimension"], categories=DIMS
    ).codes

    m = torch.tensor(model_codes, dtype=torch.long)
    q = torch.tensor(question_codes, dtype=torch.long)
    k = torch.tensor(dim_codes, dtype=torch.long)
    s = torch.tensor(data["scheme"].to_numpy(), dtype=torch.long)
    g = torch.tensor(data["grade"].to_numpy(), dtype=torch.long)

    irt = GradedMIRT(len(question_names))
    optimizer = torch.optim.Adam(irt.parameters(), lr=0.05)

    for epoch in range(epochs):
        order = torch.randperm(len(m))
        for idx in order.split(32768):
            optimizer.zero_grad()
            loss = irt(
                m[idx], q[idx], k[idx], s[idx], g[idx]
            )
            loss.backward()
            optimizer.step()

    theta = irt.theta.detach().numpy().copy()
    difficulty = irt.b.detach().numpy().copy()

    return theta, dict(zip(question_names, difficulty))



def make_features(fit_data, predict_data):
    theta, difficulty = fit_mirt(fit_data)


    items = fit_data.drop_duplicates("sample_id").copy()
    items["difficulty"] = items["sample_id"].map(difficulty)

    diff_model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=SEED,
        verbosity=-1
    )

    diff_features = ["token_length", "eval_name"]

    diff_model.fit(
        items[diff_features],
        items["difficulty"]
    )

    result = predict_data.copy()
    result["predicted_difficulty"] = diff_model.predict(
        result[diff_features]
    )

    mi = result["model_id"].map(
        MODEL_INDEX
    ).to_numpy(dtype=np.int64)

    for j, dim in enumerate(DIMS):
        result["ability_" + dim.lower()] = theta[mi, j]

    di = pd.Categorical(
        result["dimension"], categories=DIMS
    ).codes

    result["ability_current"] = theta[mi, di]
    result["ability_gap"] = (
        result["ability_current"]
        - result["predicted_difficulty"]
    )

    return result



base_features = [
    "token_length",
    "eval_name",
    "model_id"
]

ability_features = [
    "ability_math",
    "ability_code",
    "ability_general",
    "ability_current"
]

difficulty_features = [
    "predicted_difficulty"
]


gap_features = ["ability_gap"]

aug_cols = (
    ability_features
    + difficulty_features
    + gap_features
)

train_aug = train.copy()

for col in aug_cols:
    train_aug[col] = np.nan

kf = KFold(
    n_splits=3,
    shuffle=True,
    random_state=SEED
)

print("Generating out-of-fold MIRT features...")

for fold, (fit_idx, valid_idx) in enumerate(
    kf.split(train_ids), start=1
):
    fit_ids = train_ids[fit_idx]
    valid_ids = train_ids[valid_idx]

    fit_data = train[
        train["sample_id"].isin(fit_ids)
    ].copy()

    valid_data = train[
        train["sample_id"].isin(valid_ids)
    ].copy()

    print(f"Fold {fold}/3: fitting MIRT...")

    valid_aug = make_features(fit_data, valid_data)

    train_aug.loc[valid_aug.index, aug_cols] = (
        valid_aug[aug_cols].to_numpy()
    )

assert not train_aug[aug_cols].isna().any().any()


print("Fitting final MIRT on all training data...")
test_aug = make_features(train, test)


experiments = {
    "A_Baseline": base_features,

    "B_Ability": (
        base_features
        + ability_features
    ),

    "C_Difficulty": (
        base_features
        + difficulty_features
    ),

    "D_Full_MIRT": (
        base_features
        + ability_features
        + difficulty_features
        + gap_features
    )
}

results = []
predictions = test_aug[
    ["sample_id", "model_id", "eval_name", "score"]
].copy()

for name, features in experiments.items():
    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=SEED,
        verbosity=-1
    )

    model.fit(
        train_aug[features],
        train_aug["score"]
    )

    pred = np.clip(
        model.predict(test_aug[features]), 0, 1
    )

    mae = mean_absolute_error(test_aug["score"], pred)
    rmse = np.sqrt(
        mean_squared_error(test_aug["score"], pred)
    )

    results.append({
        "Experiment": name,
        "MAE": mae,
        "RMSE": rmse
    })

    predictions[name] = pred

    print(f"\n{name}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")


result_df = pd.DataFrame(results)

baseline_mae = result_df.loc[
    result_df["Experiment"] == "A_Baseline", "MAE"
].iloc[0]

baseline_rmse = result_df.loc[
    result_df["Experiment"] == "A_Baseline", "RMSE"
].iloc[0]

result_df["MAE_Improvement_%"] = (
    (baseline_mae - result_df["MAE"])
    / baseline_mae * 100
)

result_df["RMSE_Improvement_%"] = (
    (baseline_rmse - result_df["RMSE"])
    / baseline_rmse * 100
)

print("\n========== Ablation Results ==========")
print(result_df.round(6).to_string(index=False))

result_df.to_csv(
    "quality_ablation_results.csv",
    index=False
)

predictions.to_csv(
    "quality_ablation_predictions.csv",
    index=False
)

print("\nResults and predictions saved.")
