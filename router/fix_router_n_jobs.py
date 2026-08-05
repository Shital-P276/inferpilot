"""
fix_router_n_jobs.py

Permanently applies the n_jobs=1 fix to router_best_model.pkl.
Backs up the original first, then overwrites with the fixed version.

This does NOT retrain the model -- it's the exact same fitted Random Forest,
just reconfigured to use a single thread for prediction instead of spawning
one thread per core on every call. Predictions/accuracy are unaffected;
only inference speed changes.

USAGE:
    python router/fix_router_n_jobs.py
"""

import pickle
import shutil
from pathlib import Path

ROUTER_MODEL_PATH = Path("router/router_best_model.pkl")
BACKUP_PATH = Path("router/router_best_model_before_njobs_fix.pkl")


def find_n_jobs_objects(obj, seen=None):
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return []
    seen.add(id(obj))

    found = []
    if hasattr(obj, "n_jobs"):
        found.append(obj)

    for attr_name in ["steps", "transformers", "transformers_", "estimators", "estimators_"]:
        if hasattr(obj, attr_name):
            container = getattr(obj, attr_name)
            try:
                for item in container:
                    sub_obj = item[1] if isinstance(item, tuple) else item
                    found.extend(find_n_jobs_objects(sub_obj, seen))
            except TypeError:
                pass
    return found


def main():
    if not ROUTER_MODEL_PATH.exists():
        raise FileNotFoundError(f"{ROUTER_MODEL_PATH} not found. Run this from the project root.")

    # Back up the original first -- never overwrite a production artifact
    # without a way back.
    if not BACKUP_PATH.exists():
        shutil.copy(ROUTER_MODEL_PATH, BACKUP_PATH)
        print(f"Backed up original to {BACKUP_PATH}")
    else:
        print(f"Backup already exists at {BACKUP_PATH}, not overwriting it.")

    with open(ROUTER_MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    model = bundle["model"]
    n_jobs_objects = find_n_jobs_objects(model)

    if not n_jobs_objects:
        print("No n_jobs-capable objects found. Nothing changed.")
        return

    for obj in n_jobs_objects:
        old = obj.n_jobs
        obj.n_jobs = 1
        print(f"  {type(obj).__name__}: n_jobs {old} -> 1")

    with open(ROUTER_MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    print(f"\nSaved updated pickle to {ROUTER_MODEL_PATH}")
    print("Predictions/accuracy are unchanged -- only inference speed. "
          "Next step: rebuild the gateway image so it picks up this file, "
          "then re-run the core experiment to capture the updated numbers.")


if __name__ == "__main__":
    main()
