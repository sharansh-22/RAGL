import os
import subprocess
import sys
from pathlib import Path

RAGL_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_RUNNER = str(RAGL_ROOT / "evaluation" / "run.py")

MODELS = [
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "intfloat/e5-base-v2",
    "intfloat/e5-large-v2",
    "nomic-ai/nomic-embed-text-v1.5",
    "Snowflake/snowflake-arctic-embed-m-v1.5"
]

def main():
    print("=========================================")
    print("RAGL — Running Experiment A1")
    print("Objective: Embedding Model Selection")
    print("=========================================\n")

    for i, model in enumerate(MODELS, start=1):
        print(f"[{i}/{len(MODELS)}] Benchmarking model: {model}")
        
        # Check if already completed
        if str(RAGL_ROOT) not in sys.path:
            sys.path.append(str(RAGL_ROOT))
        from A.A1.config import MODEL_ALIASES
        alias = MODEL_ALIASES.get(model, "unknown")
        summary_path = RAGL_ROOT / "evaluation" / "cache" / "A1" / alias / "summary.json"
        
        if summary_path.exists():
            print(f"⏭️ Skipping {model} (already benchmarked)")
            continue
            
        env = os.environ.copy()
        env["A1_MODEL"] = model
        
        try:
            subprocess.run(
                [sys.executable, EVAL_RUNNER, "--experiment", "A1"],
                env=env,
                check=True
            )
            print(f"✅ Finished {model}\n")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed benchmarking {model}. Exit code: {e.returncode}\n")
            sys.exit(1)

    print("🎉 Experiment A1 Benchmark Phase Complete!")

if __name__ == "__main__":
    main()
