"""Execute the notebook in-place using nbconvert API."""
import sys, io
# Force UTF-8 output on Windows to avoid UnicodeEncodeError with Urdu text
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import os

nb_path = os.path.join('i23-2654_Assignment2_DS-A', 'i23-2654_Assignment2_DS-A.ipynb')
print(f"Executing notebook: {nb_path}")

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
try:
    ep.preprocess(nb, {'metadata': {'path': '.'}})
    print("Notebook executed successfully!")
except Exception as e:
    print(f"Error during execution: {str(e).encode('utf-8', 'replace').decode('utf-8')}")
    print("Saving partial results...")

with open(nb_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
print(f"Saved executed notebook to {nb_path}")
