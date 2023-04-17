# Python Data Science cheatsheet

### Data science definitions:

- Mean: Average. calculated as the sum of all values divided by the number of values. 
- Median: The middle value of the variable when sorted.
- Mode: The most frequent value in the variable.
- Trimmed Mean: The mean excluding x percent of the lowest and highest data points.
- Range: The difference between the maximum and minimum values in a variable. 
- Inter-Quartile Range (IQR): The difference between the 75th and 25th percentile values.
- Variance: The average of the squared distance from each data point to the mean.
- Standard Deviation (SD): The square root of the variance.
- Mean Absolute Deviation (MAD): The mean absolute value of the distance between each data point and the mean. (Less impacted by extreme outliers.)

### Questions to ask about a data table

- How many (non-null) observations do we have?
- How many unique columns/features do we have?
- Which columns (if any) contain missing data?
- What is the data type of each column?

### Install data science packages

```bash
pip install numpy pandas matplotlib scipy openpyxl
```
### Import data science packages

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import scipy
```

### Save and load numpy array 

Save numpy array to CSV file <br/>
`np.savetxt('fileName.csv', array)` <br/>
Load numpy array from CSV file <br/>
`array = np.loadtxt('fileName.csv')`

### Basic numpy functions

Returns sine of each value <br/>
`b = np.sin(a)` <br/>
Returns cosine of each value <br/>
`b = np.cos(a)` <br/>
Returns tangent of each value <br/>
`b = np.tan(a)` <br/>
Returns logarithm of each value <br/>
`b = np.log(a)` <br/>
Returns square root of each value <br/>
`b = np.sqrt(a)` <br/>
Returns the sum of all values <br/>
`a.sum()` <br/>
Returns the lowest value in the array <br/>
`a.min()` <br/>
Returns the highest value in the array <br/>
`a.max()` <br/>
Returns the mean of the array <br/>
`a.mean()` <br/>
Returns the median of the array <br/>
`a.median()` <br/>
Returns the standard deviation of the array <br/>
`a.std()`

### Create a DataFrame

Create a DataFrame with a dictionary
```python
data = {'column1_label':['dataC1R1', 'DataC1R2'], 'column2_label':['dataC2R1', 'dataC2R2']}
df = pd.DataFrame(data)
```
Create a DataFrame with a list
```python
data = [['dataC1R1', 'dataC2R1'], ['dataC1R2', 'dataC2R2'], ['dataC1R3', 'dataC2R3']]
df = pd.DataFrame(data, columns = ['column1_label', 'column2_label'])
```

### Open Files as DataFrame

Open CSV file <br/>
`df = pd.read_csv('fileName.csv')` <br/>
Open excel file <br/>
`df = pd.read_excel('fileName.xlsx')` <br/>
Open JSON file <br/>
`df = pd.read_json('fileName.json')` <br/>
Open HTML file <br/>
`df = pd.read_html('fileName.html')`

### Save DataFames as files

Open CSV file <br/>
`df.to_csv('fileName.csv')` <br/>
Open excel file <br/>
`df.to_excel('fileName.xlsx')` <br/>
Open JSON file <br/>
`df.to_json('fileName.json')` <br/>
Open HTML file <br/>
`df.to_html('fileName.html')` 

### Basic DataFrame functions

Show the first few rows of the table. <br/>
`print(df.head(n))` <br/>
Where n is the first number of rows to show 

Show general information about the table. <br/>
`print(df.info())` 

Show the datatypes of the table <br/>
`print(df.dtypes)`

Show the number of dimensions of the table <br/>
`print(df.ndim)`

Show the number of elements of the table. <br/>
`print(df.size)`

Get the number of elements in a column. <br/>
`column_size = len(df.column_name)`

Fill values in a column that are NaN or None with a different value <br/>
`df['column_name'].fillna(x, inplace = True)` <br/>
Where x is a value. Note that the value should be the same datatype as the other values in the column.

Cast the datatypes in a column to a different datatype. <br/>
`df['column_name'] = df['column_name'].astype('dataType')` <br/>

Cast a column to a category with an order <br/>
`df['column_name'] = pd.Categorical(df['column_name'], ['value1', 'value2', 'value3'], ordered=True)`

Create a listing of how many times each value in a column appears. Ordered from high to low. (Returns a series) <br/>
`counted_values = df['column_name'].value_counts()` <br/>
or add normalize=True to get the percentages of how many each value appears in the column. <br/>
`counted_values_proportion = df['column_name'].value_counts(normalize=True)`

Create a table which includes both the counted values of a column and its frequency in percentage. 
```python
counted_values = df['column_name'].value_counts()
counted_values_proportion = df['column_name'].value_counts(normalize=True)
counted_values_proportion = round(counted_values_proportion, 4) * 100
count_table = pd.DataFrame({'counts': counted_values, 'percentages': counted_values_proportion})
```

### Aggregate functions:

- `.sum()`                       Returns the sum of values in column.
- `.mean()` 	                 Average of all values in column.
- `.median()` 	                 Median.
- `.mode()`                      Returns the value that occurs most often in the column.
- `.std()` 	                     Standard deviation.
- `.mad()`                       Mean absolute deviation.
- `.var()`                       Variance.
- `.max()` 	                     Maximum value in column.
- `.min()` 	                     Minimum value in column.
- `.abs()`                       Returns the absolute values of the elements.
- `.count()`  	                 Number of values in column.
- `.nunique()` 	                 Number of unique values in column.
- `.unique()` 	                 List of unique values in column.
- `.prod()`                      Returns product of selected elements.
- `.describe(include = 'all')`   Returns data frame with all statistical values summarized.

Aggregate function syntax: <br/>
`df.column_name.command()`

Calculate aggregates: <br/>
`df.groupby('column1').column2.measurement()` <br/>
Turn the datatype from series to a dataframe with new indices. <br/>
`df.groupby('column1').column2.measurement().reset_index()` <br/>
Group by multiple columns. <br/>
`df.groupby(['column1', 'column2']).column3.measurement().reset_index()`

- column1 is the column that we want to group by.
- column2 is the column that we want to perform a measurement on. 
- measurement is the measurement function we want to apply.

### Change column name of dataframe:

`df = df.rename(columns={"old_column_name": "new_column_name"})`

### Add new column to dataframe

Applies single value to all rows. <br/>
`df['column_name'] = "Hello world"` 

Give individual manual values. <br/>
`df['column_name'] = [0,1,2,3,4,5,6]` 

Applies value based on other row value (multiply value by 2 in this case). <br/>
`df['column_name'] = df.column_name2 * 2`

### Sorting data frames

Sort dataFrame by values of a column_name by descending and putting the NaN values last. (Returns new DataFrame) <br/>
`sorted_df = df.sort_values(by='column_name', ascending=False, na_position='last')`

Sort column by values by ascending and putting the NaN values first.  (Returns Series) <br/>
`sorted_column = df['column_name'].sort_values(ascending=True, na_position='first')`

### Apply lambda function to data frame

`df['column_name'] = df.apply(lambda x: True if x=True, else False axis = 1)`

### Data frame filtering

Creates a new data frame with the values of column filtered based on the conditional. <br/>
`new_df = df[df['column_name'] conditional_operator value]` <br/>
Replace conditional_operator with >, <, ==, !=, etc.

### Create pivot table

```python
pivotTable = df.pivot(columns='ColumnToPivot',
         index='ColumnToBeRows',
         values='ColumnToBeValues').reset_index()
```

### Merge DataFrames (tables) together

Merge two tables together. <br/>
`new_df = pd.merge(df1, df2)` <br/>
or <br/>
`new_df = df1.merge(df2)` 

Merge 2 tables with rename of columns. <br/>
`new_df = pd.merge(df1, df2.rename(columns={'old_column_name': 'new_column_name'}))`

Merge 2 tables with suffixes.
```python
new_df = pd.merge(
    df1,
    df2,
    left_on='column_in_df1',
    right_on='column_in_df2',
    suffixes=['_df1', '_df2']
)
```

Merge 2 tables with outer join. (Includes mismatching rows.) Missing data will be NaN or None. <br/>
`new_df = pd.merge(df1, df2, how='outer')`

Merge 2 tables with left join. (Includes all rows from df1, and only rows from df2 that match with df1) <br/>
`pd.merge(df1, df2, how='left')`

Merge 2 tables with right join. (Includes all rows from df2, and only rows from df1 that match with df2) <br/>
`pd.merge(df1, df2, how='right')`

Concatenate multiple DataFrames. This only works if all dataframes have the exact same columns <br/>
`new_df = pd.concat([df1, df2, df3, ...])`

### One-Hot Encode a DataFrame

Create binary variables of each category from the values in column_name <br/>
`new_df = pd.get_dummies(df, columns=['column_name'])`

### Calculate common statistics

Calculate the trimmed mean <br/>
```python
from scipy.stats import trim_mean
# Replace value with a float number between 0 and 1.
# For example 0.1 will trim 10% off the extremes.
trmean = trim_mean(df.column_name, proportiontocut=value)
```

Calculate the range: <br/>
`min_max_range = df.column_name.max() - df.column_name.min()`

Calculate the Interquartile range (IQR): <br/>
`iqr_value = df.column_name.quantile(0.75) - df.column_name.quantile(0.25)` <br/>
or 
```python
from scipy.stats import iqr
iqr_value = iqr(df.column_name) 
```

Create a covariance matrix between 2 variables. <br/>
A covariance of 0 indicates no relationship between the variables. <br/>
`cov_mat = np.cov(df.column_one, df.column_two)`

Calculate the correlation between 2 variables. <br/>
A value larger than .3 shows a linear association. <br/>
A value larger than .6 shows a very strong linear association.
```python
from scipy.stats import pearsonr
correlation, p = pearsonr(df.column_one, df.column_two)
```

Create a contingency table of frequencies between two categorical columns <br/>
The contingency table shows the amount of times(counts) each combination of categories appears. <br/>
`cont_freq = pd.crosstab(df.column_one, df.column_two)`

Create a contingency table, using proportions(percentages) instead of counts.
```python
cont_freq = pd.crosstab(df.column_one, df.column_two)
cont_prop = influence_leader_freq / len(df.column_one)
```

### Data visualization

Create a boxplot from a column in a dataframe
```python
sns.boxplot(x='column_name', data=df)
plt.show()
plt.close()
```

Create a side by side boxplot from a column vs another column in a dataframe. <br/>
x is usually a category while y is a numerical data. <br/>
boxplots that overlap show a weak association between the values and the categories.
```python
sns.boxplot(data = df, x = 'column_one', y = 'column_two')
sns.boxplot(x='column_name', data=df)
plt.show()
plt.close()
```

Create a histogram from a column in a dataframe
```python
sns.histplot(x='column_name', data=df)
plt.show()
plt.close()
```

Create two histograms on top of each other from two series. <br/>
Alpha sets the transparency for each histogram.
```python
plt.hist(series_one , color="blue", label="series one", density=True, alpha=0.5)
plt.hist(series_two , color="red", label="series two", density=True, alpha=0.5)
plt.legend()
plt.show()
plt.close()
```

Create a bar chart from a column in a dataframe. The barchart visualises the counts of each value in the column.
```python
sns.countplot(x='column_name', data=df)
plt.show()
plt.close()
```

Create a bar chart from a column in a dataframe. The piechart visualises the counts of each value in the column.
```python
df.column_name.value_counts().plot.pie()
plt.show()
plt.close()
```

Create a scatter plot to see a relationship between two columns. The values in both columns are usually continuous and non-categorical.
```python
plt.scatter(x = df.column_one, y = df.column_two)
plt.xlabel('column one')
plt.ylabel('column two')
plt.show()
```