from parser import ClevrParser
import pandas as pd
import os
from tqdm import tqdm

folder_name = "results/gpt-4o/clevr"
file_name_pattern = "results_{}.csv"

# loop through all files in the folder and test if the models in the file can be parsed
for idx in tqdm(range(1, 21), desc="Parsing activity graphs"):
    filename = file_name_pattern.format(idx)
    file_path = os.path.join(folder_name, filename)
    df = pd.read_csv(file_path)
    graphs = df["0"].tolist()
    parser = ClevrParser()
    for i, mermaid_text in enumerate(
        tqdm(graphs, desc=f"File {filename}", leave=False)
    ):
        try:
            graph = parser.parse(mermaid_text)
        except Exception as e:
            print(f"Error parsing graph in file {filename}, row {i}: {e}")
            raise e

filename = file_name_pattern.format("greedy")
file_path = os.path.join(folder_name, filename)
df = pd.read_csv(file_path)
graphs = df["0"].tolist()
parser = ClevrParser()
for i, mermaid_text in enumerate(
    tqdm(graphs, desc=f"File {filename}", leave=False)
):
    try:
        graph = parser.parse(mermaid_text)
    except Exception as e:
        print(f"Error parsing graph in file {filename}, row {i}: {e}")
        raise e

print("Parsing completed.")
