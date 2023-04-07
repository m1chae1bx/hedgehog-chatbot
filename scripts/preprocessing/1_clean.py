import json
import os
import pandas as pd

RAW_DATA_PATH = "data/raw/financial_news/json"
OUTPUT_DIR = "data/processed/cleaned_json"

df = pd.DataFrame()
counter = 0
file_number = 1


def save_df_to_json(df, file_number, output_dir):
    output_path = os.path.join(output_dir, f"processed_{file_number:04d}.json")
    df.to_json(output_path, orient="records", lines=True)


def prepare_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            os.remove(os.path.join(output_dir, file))


prepare_output_dir(OUTPUT_DIR)

for file in os.listdir(RAW_DATA_PATH):
    if file.endswith(".json"):
        with open(os.path.join(RAW_DATA_PATH, file), "r") as f:
            data = json.load(f)
            row = {
                "published": data["published"],
                "title": data["title"],
                "text": data["text"],
            }
            df = pd.concat(
                [df, pd.DataFrame(row, index=[0])],
                ignore_index=True,
            )
            counter += 1

        if counter >= 1000:
            save_df_to_json(df, file_number, OUTPUT_DIR)
            df = pd.DataFrame()
            counter = 0
            file_number += 1

if not df.empty:
    save_df_to_json(df, file_number, OUTPUT_DIR)
