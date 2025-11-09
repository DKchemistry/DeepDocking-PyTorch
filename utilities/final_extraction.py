from multiprocessing import Pool
from contextlib import closing
import multiprocessing
import pandas as pd
import argparse
import glob
import os


def merge_on_smiles(pred_file):
    print("Merging " + os.path.basename(pred_file) + "...")

    # Read the predictions: CSV, no header, id,score
    pred = pd.read_csv(pred_file, names=["id", "score"])
    pred = pred.drop_duplicates()

    # Matching smiles file has the same basename, space-delimited: smile id
    smile_file = os.path.join(args.smile_dir, os.path.basename(pred_file))
    smi = pd.read_csv(smile_file, delimiter=" ", names=["smile", "id"])
    smi = smi.drop_duplicates()

    merged = pd.merge(pred, smi, how="inner", on=["id"]).set_index("id")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-smile_dir", required=True, help="Path to SMILES directory for the database"
    )
    parser.add_argument(
        "-prediction_dir",
        required=True,
        help="Path to morgan_1024_predictions of last iteration",
    )
    parser.add_argument(
        "-processors", required=True, help="Number of CPUs for multiprocessing"
    )
    parser.add_argument(
        "-mols_to_dock",
        required=False,
        type=int,
        help="Desired number of molecules to dock",
    )
    parser.add_argument(
        "-output_dir",
        required=True,
        help="Directory where smiles.csv and id_score.csv will be written",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    predictions = []

    print("Prediction dir: " + args.prediction_dir)
    print("SMILES dir: " + args.smile_dir)

    # discover prediction files
    for file in glob.glob(os.path.join(args.prediction_dir, "*")):
        # keep the old heuristic: only files that have "smile" in their name
        if "smile" in os.path.basename(file):
            print(" - " + os.path.basename(file))
            predictions.append(file)

    if not predictions:
        raise RuntimeError(
            "No prediction files found matching pattern '*smile*' in "
            f"{args.prediction_dir}"
        )

    print("Finding smiles...")
    num_jobs = min(len(predictions), int(args.processors))
    print(
        f"Using {num_jobs} workers (requested {args.processors}, found {len(predictions)} files)"
    )
    print("Machine has", multiprocessing.cpu_count(), "CPUs")

    with closing(Pool(num_jobs)) as pool:
        combined = pool.map(merge_on_smiles, predictions)

    # combine
    print("Combining", len(combined), "dataframes...")
    base = pd.concat(combined, axis=0)
    combined = None

    print("Sorting by score (descending)...")
    base = base.sort_values(by="score", ascending=False)
    base.reset_index(inplace=True)

    # optional truncation
    if args.mols_to_dock is not None:
        mtd = args.mols_to_dock
        print("Molecules to dock:", mtd)
        print("Total molecules:", len(base))
        if len(base) > mtd:
            print(f"Keeping top {mtd} molecules")
            base = base.head(mtd)
        else:
            print("Total <= requested, keeping all")

    # write smiles
    print("Saving outputs to:", args.output_dir)

    smiles = base.drop("score", axis=1)
    smiles = smiles[["smile", "id"]]
    smiles_path = os.path.join(args.output_dir, "smiles.csv")
    smiles.to_csv(smiles_path, sep=" ", index=False)

    # write id-score
    id_score = base.drop("smile", axis=1)
    id_score_path = os.path.join(args.output_dir, "id_score.csv")
    id_score.to_csv(id_score_path, index=False)

    print("Sample smiles:")
    print(smiles.head())
    print("Sample id-score:")
    print(id_score.head())
    print("Done.")
