import pandas as pd

# Load the Excel file
df = pd.read_excel('Data1P_05_105_06_106.xlsx', index_col=0)

# # Transpose the DataFrame
df_transposed = df.T

df_transposed.columns = df_transposed.iloc[0]

# Drop the row that was used for column names and reset the index
df_transposed = df_transposed.iloc[1:].reset_index(drop=True)

# Delete the Female gender column column
df_transposed = df_transposed.loc[:, df_transposed.columns.notna()]
df_transposed = df_transposed.iloc[2:]

# 8. Transposed Data
existing_excel_file = 'Transposed_Data.xlsx'
df_transposed.to_excel(existing_excel_file, index=False)

# 1. Impute Missing Values: We'll fill missing values with the 0 as all columns are numerical
df_transposed.fillna(0, inplace=True)

# 2. Convert heights to inches (only if they are not already in inches)
df_transposed['What is your height in inches?'] = df_transposed['What is your height in inches?'].apply(lambda x: x * 12 if x < 12 else x)  # Convert feet to inches if value is less than 12

# 3. Ensure age is in months
df_transposed['What is your age in # of months?'] = df_transposed['What is your age in # of months?'].apply(lambda x: x * 12 if x < 100 else x) # Convert years to months if value is less than 100

# 4. Ensure weight is in pounds
df_transposed['What is your weight in pounds?'] = df_transposed['What is your weight in pounds?'].apply(lambda x: x * 2.205 if x < 90 else x) # Convert kilograms to pounds if value is less than 90

# 5. Ensure that the salary is in USD
# No Salary Column in the excel file

# 6. Ensure that the prior work experience is in months
df_transposed['How many months of paid experience did you have before you started your graduate degree at UTA?'] = df_transposed['How many months of paid experience did you have before you started your graduate degree at UTA?'].apply(lambda x: x * 12 if x < 5 else x) # Convert years to months if value is less than 5

df_transposed['How many months of paid experience did you have before you started your graduate degree at UTA?'] = df_transposed['How many months of unpaid experience did you have before you started your graduate degree at UTA?'].apply(lambda x: x * 12 if x < 2 else x) # Convert years to monthsif value is less than 2

# 7. Rectify incorrect data
# (A) Missing Values
#     Problem: Some records have missing values.
#     Solution: Use imputation methods (mean, median, mode, or advanced techniques) to fill in missing values. Choose the imputation method based on data characteristics and the    impact on analysis.
#     Explanation: Missing values can lead to biased analysis and loss of information. Imputation helps maintain data integrity.

# (B) Inconsistent Units:

#   Problem: Values in the dataset might be in different units or scales.
#   Solution: Convert all values to a consistent unit or scale, ensuring that all data are comparable.
#   Explanation: Inconsistent units can lead to misinterpretation and incorrect analysis. Standardizing units is crucial for accuracy.

# (C) Missing Columns or Incorrect Metadata:

#   Problem: Metadata (column descriptions, units, etc.) may be missing or incorrect.
#   Solution: Update metadata to ensure accurate understanding and interpretation of the dataset.
#   Explanation: Accurate metadata is crucial for understanding the dataset and its context.

# 8. Rename Columns
df_transposed.rename(columns = {'What is your height in inches?':'Height'}, inplace = True)
df_transposed.rename(columns = {'What is your age in # of months?':'Age(Months)'}, inplace = True)
df_transposed.rename(columns = {'What is your weight in pounds?':'Weight'}, inplace = True)
df_transposed.rename(columns = {'What is your gender':'Gender(M/F)'}, inplace = True)
df_transposed['Gender(M/F)'] = df_transposed['Gender(M/F)'].replace({1: 'M', 0: 'F'})
df_transposed['Experience(Months)'] = df_transposed['How many months of paid experience did you have before you started your graduate degree at UTA?'] + df_transposed['How many months of unpaid experience did you have before you started your graduate degree at UTA?']
df_transposed.drop(['How many months of paid experience did you have before you started your graduate degree at UTA?', 'How many months of unpaid experience did you have before you started your graduate degree at UTA?'], axis=1, inplace=True)

df_transposed.to_csv('ClassData.csv', index=False)







