import argparse
import subprocess
import sys

sys.dont_write_bytecode = True
from common import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20000)
    args = parser.parse_args()
    commands = [["prepare_data.py", "--max-requests", str(args.rows)],
                ["analyze_benchmarks.py"], ["validate_data.py"],
                ["run_experiments.py"], ["validate.py"],
                ["validate_results.py"], ["write_report.py"]]
    for step, command in enumerate(commands, 1):
        print("STEP %d/%d: %s" % (step, len(commands), " ".join(command)), flush=True)
        subprocess.run([sys.executable, "-B", str(ROOT / command[0]), *command[1:]], cwd=str(ROOT), check=True)
    print("Complete. See outputs/run_manifest.json and README_zh.md.")


if __name__ == "__main__":
    main()
