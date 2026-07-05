"""
RAGL — A2 Experiment Runner
===========================
Iterates over the 4 chunking strategies, overriding A2_CHUNKER_STRATEGY,
and invokes the benchmark runner for each.
"""

import os
import subprocess
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RAGL_ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_RUNNER = RAGL_ROOT / "evaluation" / "run.py"

STRATEGIES = [
    "recursive",
    "sentence",
    "structure",
    "semantic"
]

def run_strategy(strategy: str):
    logger.info("=" * 60)
    logger.info(f"🚀 STARTING A2 EXPERIMENT FOR STRATEGY: {strategy}")
    logger.info("=" * 60)

    env = os.environ.copy()
    env["A2_CHUNKER_STRATEGY"] = strategy

    cmd = [
        sys.executable,
        str(EVAL_RUNNER),
        "--experiment", "A2",
        "--datasets", "all",
        "--rebuild"
    ]

    try:
        subprocess.run(cmd, env=env, check=True)
        logger.info(f"✅ Completed benchmark for strategy: {strategy}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Benchmark failed for strategy {strategy} with exit code {e.returncode}")
        sys.exit(e.returncode)

def main():
    logger.info("Starting Experiment A2: Chunking Strategy Selection")
    
    for strategy in STRATEGIES:
        run_strategy(strategy)

    logger.info("🎉 All A2 chunking strategies have been benchmarked!")
    logger.info("Please run `python evaluation/compare.py` to generate the final study report.")

if __name__ == "__main__":
    main()
