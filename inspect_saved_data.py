
import joblib
import yaml
from pathlib import Path

split_folder = Path("data/interim/validation_splits")

for name in ["nsl_kdd", "unsw_nb15"]:
    file_path = split_folder / f"{name}_split.joblib"

    data = joblib.load(file_path)

    print("\n" + "=" * 55)
    print(name.upper())
    print("=" * 55)

    print("Object type:", type(data))

    if isinstance(data, dict):
        print("Keys:", list(data.keys()))

        for key, value in data.items():
            print(
                key,
                "| Type:", type(value).__name__,
                "| Shape:", getattr(value, "shape", None)
            )
    else:
        print("Contents:", data)


descriptor_path = Path(
    "src/novel_ids/onboarding/feature_descriptors_with_stats.yaml"
)

with open(descriptor_path, "r", encoding="utf-8") as file:
    descriptors = yaml.safe_load(file)

print("\n" + "=" * 55)
print("FEATURE DESCRIPTOR STRUCTURE")
print("=" * 55)

print("Top-level keys:", list(descriptors.keys()))
print("Total descriptors:", len(descriptors["features"]))
print("First descriptor:")
print(descriptors["features"][0])
