import json
import os
import logging
import sys


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

import config
from src.processing.cleaner import clean_code
from src.processing.tagger import tag_code_blocks
from src.processing.normalizer import normalize_code

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_json(file_path):

    logging.info(f"Loading file: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    with open(file_path, "r") as f:
        return json.load(f)


def has_class_def(code):
    return "class " in code


def create_initial_dataset(raw_path, complexity_path):
    solution_id_to_complexity = load_json(complexity_path)
    if solution_id_to_complexity is None:
        return []

    final_dataset = []
    with open(raw_path, "r") as f:
        for line in f:
            data = json.loads(line)
            for sol in data.get("correct_solution_list", []):
                solution_id = sol.get("solution_id")
                solution_code = sol.get("solution_code")
                if solution_id in solution_id_to_complexity and solution_code:
                    cleaned_code = clean_code(solution_code)
                    if cleaned_code:
                        entry = {
                            "solution_code": cleaned_code,
                            "time_complexity": solution_id_to_complexity[solution_id]
                        }
                        final_dataset.append(entry)
    logging.info(f"Created {len(final_dataset)} examples.")
    return final_dataset


def main():

    dataset = create_initial_dataset(config.RAW_DATA_PATH, config.COMPLEXITY_MAPPING_PATH)

    logging.info("Additional filter...")
    dataset = [e for e in dataset if e.get("solution_code", "").strip()]
    dataset = [e for e in dataset if not has_class_def(e["solution_code"]) and "__main__" not in e["solution_code"]]
    logging.info(f"The dataset contains {len(dataset)} example after the filtering.")

    logging.info("Normalization and tagging in progress...")
    for entry in dataset:
        normalized_code = normalize_code(entry["solution_code"])
        tagged_code = tag_code_blocks(normalized_code)
        entry["solution_code"] = tagged_code

    logging.info(f"Saving the final dataset in {config.PROCESSED_DATASET_PATH}")
    os.makedirs(os.path.dirname(config.PROCESSED_DATASET_PATH), exist_ok=True)
    with open(config.PROCESSED_DATASET_PATH, "w") as f:
        json.dump(dataset, f, indent=4)

    logging.info("Process completed!")


if __name__ == "__main__":
    main()