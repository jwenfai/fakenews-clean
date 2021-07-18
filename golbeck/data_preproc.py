from pathlib import Path
import pandas as pd
import chardet
import os

golbeck_dir = Path("K:/Work/Datasets-FakeNews/source-reliability/golbeck/FakeNewsData")
fake_dir = golbeck_dir / "StoryText 2/Fake/finalFake"
satire_dir = golbeck_dir / "StoryText 2/Satire/finalSatire"
fakes_df_path = golbeck_dir / "fakes_df.tsv"
satires_df_path = golbeck_dir / "satires_df.tsv"

fakes_txt = []
for file in os.listdir(fake_dir):
    if file.endswith(".txt"):
        encoding = chardet.detect(open(fake_dir/file, "rb").read())['encoding']
        with open(fake_dir/file, "r", encoding=encoding) as f:
            all_lines = f.readlines()
            title = all_lines[0].rstrip("\n")
            url = all_lines[1].rstrip("\n")
            body = " \n ".join(all_lines[2:]).rstrip("\n")
            fakes_txt.append([title, url, body])

satires_txt = []
for file in os.listdir(satire_dir):
    if file.endswith(".txt"):
        encoding = chardet.detect(open(satire_dir/file, "rb").read())['encoding']
        with open(satire_dir/file, "r", encoding=encoding) as f:
            all_lines = f.readlines()
            title = all_lines[0]
            url = all_lines[1]
            body = " \n ".join(all_lines[2:])
            satires_txt.append([title, url, body])

col_names = ["title", "url", "body"]
fakes_df = pd.DataFrame(fakes_txt, columns=col_names)
satires_df = pd.DataFrame(satires_txt, columns=col_names)

fakes_df.to_csv(fakes_df_path, sep="\t", index=False)
satires_df.to_csv(satires_df_path, sep="\t", index=False)

# Check if data has been altered from writing to CSV
temp = pd.read_csv(fakes_df_path, sep="\t").fillna("")
