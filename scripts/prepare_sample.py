import argparse
from pathlib import Path
import pandas as pd

parser = argparse.ArgumentParser(description='Prélever un échantillon chronologique du fichier Sparkov')
parser.add_argument('csv', type=Path, help='Chemin vers fraudTrain.csv')
parser.add_argument('--rows', type=int, default=25000)
parser.add_argument('--output', type=Path, default=Path('sparkov_sample.csv'))
args = parser.parse_args()
chunks = []
count = 0
for chunk in pd.read_csv(args.csv, chunksize=10000):
    chunks.append(chunk)
    count += len(chunk)
    while count > args.rows + 10000 and len(chunks) > 1:
        count -= len(chunks.pop(0))
out = pd.concat(chunks, ignore_index=True).tail(args.rows)
out.to_csv(args.output, index=False)
print(f'Échantillon créé : {args.output} ({len(out)} lignes)')
