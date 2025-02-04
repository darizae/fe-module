import json
from pathlib import Path
from dataclasses import dataclass
from device_selector import check_or_select_device

from fenice_custom import FENICECustomClaims


@dataclass
class RosePaths:
    # This example matches your pattern:
    BASE_DIR = Path(__file__).resolve().parent
    dataset_path: Path = BASE_DIR / "rose_datasets_small.json"
    results_dir: Path = BASE_DIR / "fenice_results"


def load_rose_dataset(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # 1. Select device using device-selector
    device = check_or_select_device(None)  # or "cuda", "mps", etc.

    # 2. Prepare directories
    paths = RosePaths()
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    # 3. Load the full RoSE data
    rose_data = load_rose_dataset(paths.dataset_path)
    # Suppose the top-level keys in rose_data are:
    # ["cnndm_test", "cnndm_validation", "xsum", "samsum"]

    # 4. For demonstration, process "cnndm_test" subset
    subset_key = "cnndm_test"
    records = rose_data[subset_key]  # This is a list of records

    all_scores = []
    # Loop over each record in that subset
    for idx, record in enumerate(records):
        doc = record["source"]
        summary = record["reference"]
        record_id = record.get("record_id", f"rec_{idx}")

        # unify reference_acus claims + system_claims into one dictionary
        all_claim_sets = {}

        # 1) reference_acus
        reference_acus = record.get("reference_acus", {})
        # If "deduped_0.7_select_longest" exists, treat it as one “system-like” set
        if "deduped_0.7_select_longest" in reference_acus:
            all_claim_sets["deduped_0.7_select_longest"] = reference_acus["deduped_0.7_select_longest"]
        # If you want to handle "original" or other variants, just add them:
        # all_claim_sets["original"] = reference_acus["original"]

        # 2) system_claims
        system_claims = record.get("system_claims", {})
        # This might have multiple model names
        for model_name, claims in system_claims.items():
            all_claim_sets[model_name] = claims

        # Now evaluate each claim set
        for claim_set_name, claims_list in all_claim_sets.items():
            # Prepare the custom_claim_map
            summary_id = f"{idx}{summary[:100]}"  # mimic FENICE’s get_id
            custom_claim_map = {summary_id: claims_list}

            # Instantiate the FENICE subclass
            fenice_evaluator = FENICECustomClaims(
                custom_claims_by_summary_id=custom_claim_map,
                device=device,
                use_coref=False
            )

            # Evaluate on a single batch (one doc-summary pair)
            batch = [{"document": doc, "summary": summary}]
            result = fenice_evaluator.score_batch(batch)
            score = result[0]["score"]  # overall factuality score

            # Save or print
            record_result = {
                "subset_key": subset_key,
                "record_id": record_id,
                "claim_set_name": claim_set_name,
                "score": score,
                "alignments": result[0]["alignments"]
            }
            all_scores.append(record_result)

    # Do whatever you want with all_scores: print it, write to JSON, etc.
    print(all_scores)


if __name__ == "__main__":
    main()
