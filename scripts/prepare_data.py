import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_DIR / "data" / "linuxcommands.json"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


df = pd.read_json(DATASET_PATH)
df = df.dropna().drop_duplicates().reset_index(drop=True)


train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42
)



train_df.to_json(OUTPUT_DIR / "train.json", orient="records", indent=2)
val_df.to_json(OUTPUT_DIR / "val.json", orient="records", indent=2)
test_df.to_json(OUTPUT_DIR / "test.json", orient="records", indent=2)