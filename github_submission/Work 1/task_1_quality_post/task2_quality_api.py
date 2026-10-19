from pathlib import Path
import importlib.util
import json
import joblib
import numpy as np
import pandas as pd







HERE = Path(__file__).resolve().parent


SAVE = HERE / "saved_models"


BASE_SCRIPT = HERE / "quality(sub-model_assessment_aware_ablation).py"






if not BASE_SCRIPT.exists():
    raise FileNotFoundError(
        f"找不到基础评价脚本：\n{BASE_SCRIPT}\n"
        "请检查文件名是否与PyCharm中的实际名称一致。"
    )

spec = importlib.util.spec_from_file_location(
    "quality_base",
    str(BASE_SCRIPT)
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"无法加载基础评价脚本：{BASE_SCRIPT}"
    )

base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)






class PostQualityAssessor:

    def __init__(self, model_dir=SAVE):

        model_dir = Path(model_dir)

        required_files = [
            "metadata.json",
            "V.joblib",
            "VC_2.joblib"
        ]

        for filename in required_files:
            filepath = model_dir / filename

            if not filepath.exists():
                raise FileNotFoundError(
                    f"找不到模型文件：{filepath}\n"
                    "请先运行export_assessor脚本训练并保存模型。"
                )


        self.meta = json.loads(
            (model_dir / "metadata.json").read_text(
                encoding="utf-8"
            )
        )


        self.v = joblib.load(
            model_dir / "V.joblib"
        )


        self.vc2 = joblib.load(
            model_dir / "VC_2.joblib"
        )

        print("PostQualityAssessor加载成功！")


    def assess(
        self,
        prompt,
        response,
        model_id,
        task_kind="multiple_choice",
        peer_response=None
    ):


















        if model_id not in self.meta["models"]:
            raise ValueError(
                f"未知模型ID：{model_id}\n"
                "请使用训练数据中的模型名称。"
            )





        answer = base.norm_answer(response)

        z = base.diagnostic(
            str(prompt),
            answer,
            task_kind
        )


        out = {
            "task_kind": task_kind,
            "hard_flag": bool(z["hard_flag"]),
            "format_error": bool(z["format_error"]),
            "correctness_verified": False,
            "quality_proxy": None,
            "method": "diagnostics_only"
        }





        if task_kind != "multiple_choice":
            return out





        z["model_id"] = model_id

        z["peer_valid"] = 0
        z["peer_agree"] = np.nan


        if peer_response is not None:

            peer_answer = base.choice(
                base.norm_answer(peer_response)
            )

            z["peer_valid"] = int(
                peer_answer is not None
            )

            if (
                z["answer_choice"] is not None
                and peer_answer is not None
            ):
                z["peer_agree"] = float(
                    z["answer_choice"] == peer_answer
                )


        row = pd.DataFrame([z])

        row["model_id"] = pd.Categorical(
            row["model_id"],
            categories=self.meta["models"]
        )





        if peer_response is None:

            features = self.meta["valid_features"]

            pred = self.v.predict(
                row[features]
            )[0]

            out["quality_proxy"] = float(
                np.clip(pred, 0, 1)
            )

            out["method"] = "V"





        else:

            features = self.meta["two_features"]

            pred = self.vc2.predict(
                row[features]
            )[0]

            out["quality_proxy"] = float(
                np.clip(pred, 0, 1)
            )

            out["method"] = "VC_2"

            out["peer_valid"] = bool(
                z["peer_valid"]
            )

            out["peer_agree"] = (
                float(z["peer_agree"])
                if pd.notna(z["peer_agree"])
                else None
            )

            out["warning"] = (
                "VC_2验证时采用固定第二模型规则，"
                "动态选择第二模型需要重新校准。"
            )

        return out






if __name__ == "__main__":

    assessor = PostQualityAssessor()


    model_id = assessor.meta["models"][0]

    prompt = "Choose the correct answer: A, B, C or D"

    response = "A"

    print("\n========== 单模型评价 ==========")

    result_v = assessor.assess(
        prompt=prompt,
        response=response,
        model_id=model_id,
        task_kind="multiple_choice"
    )

    print(result_v)

    print("\n========== 双模型评价 ==========")

    result_vc2 = assessor.assess(
        prompt=prompt,
        response=response,
        model_id=model_id,
        task_kind="multiple_choice",
        peer_response="A"
    )

    print(result_vc2)

    print("\n========== 非选择题评价 ==========")

    result_other = assessor.assess(
        prompt="Write a Python function",
        response="def add(a, b): return a + b",
        model_id=model_id,
        task_kind="code"
    )

    print(result_other)
