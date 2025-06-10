import pandas as pd
from tabulate import tabulate

# Load the XLSX file
file_path = 'Models.xlsx' # Replace with your file path
df = pd.read_excel(file_path)

# Convert the DataFrame to a LaTeX table
latex_table = tabulate(df, tablefmt="latex", headers="keys",showindex=False)

# Print the LaTeX table
print(latex_table)

# Save the LaTeX table to a file
with open('models.tex', 'w') as f:
    f.write(latex_table)