from datasets import load_dataset
import os, json

def load_GAIA_dataset(dataset_name, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ds = load_dataset(dataset_name, "2023_all", split="validation")
    data = ds.to_list()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} items to {output_path}")
    return data

def get_GAIA_dataset():
    output_path = "../datasets/GAIA/data.json"

    if not os.path.exists(output_path):
        data = load_GAIA_dataset("gaia-benchmark/GAIA", output_path)
    else:
        # load from local JSON
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} items from {output_path}")
    return data



if __name__ == "__main__":
    get_GAIA_dataset()