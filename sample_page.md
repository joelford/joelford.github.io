## LendingClub Data Exploration and Modelling

**Project description:** LendingClub is an American peer-to-peer company, enabling users to create unsecured personal loans between $1000 and $40 000 with a standard period of three years. Investors are able to search and browse the loan listings on LendingClub website and select loans that they want to invest in based on the information supplied about the borrower, amount of loan, loan grade, and loan purpose. (https://en.wikipedia.org/wiki/LendingClub)
There are two primary goals to this project:
1. To build a system to quickly explore and visualize the data, comparing features to the target loan status
2. To model the data in various ways to determine which method best predicts if a loan is charged off, using only information available prior to loan origination. Such a model could help potential investors quickly consolidate information and make a better informed decision. 

### 1. Importing the Data and Cleaning

Data available from: https://www.kaggle.com/wordsforthewise/lending-club/

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as spstats
import seaborn as sns
import re
```

Reading the data into a Pandas dataframe

```python
loan = pd.read_csv('data/accepted_2007_to_2018q4/accepted_2007_to_2018Q4.csv', low_memory=False)
```

Analysing how much missing data we have

```python
missing = loan.isnull().mean().sort_values(ascending=False)
print(missing)
missing.plot.hist(bins=20)
```
<img src="images/missing.png?raw=true"/>

Many columns are nearly or completely empty. Only keep columns with >30% data
```python
drop = missing[missing > 0.30].index
loan.drop(drop, axis=1, inplace=True)
```

Get the columns we currently have kept, save to a file for reference
```python
import os.path
def columns(df, filename, to_print=False):
    columns_sorted = df.columns.sort_values()
    
    if not os.path.isfile(filename): # Only make this file once
        with open(filename, 'a') as f:
            for col in columns_sorted:
                if to_print:
                    print(col)
                    
                f.write(col+'\n')
                
        f.close()
    
    else:
        print(filename+' already exists.')

column(loan, 'loan_columns_missing_removed.txt')
```

Remove irrelevant/undesirable columns. Information on variable defintions: https://resources.lendingclub.com/LCDataDictionary.xlsx.
Reasons for removal:
1. Irrelevant (ex: id)
2. Information was gained after loan origination (ex: last payment)
```python
remove = ['collection_recovery_fee',  # 2
            'collections_12_mths_ex_med',  # 2
            'debt_settlement_flag',  # 2
            'disbursement_method',  # 1
            'earliest_cr_line',  # 1
            'emp_title', # Far too many categories, drop
            'funded_amnt',  # 2
            'funded_amnt_inv',  # 2
            'hardship_flag',  # 2
            'id',  # 1
            'initial_list_status',  # 1
            'last_credit_pull_d',  # 1
            'last_fico_range_high',  # 2
            'last_fico_range_low',  # 2
            'last_pymnt_d',  # 2
            'last_pymnt_amnt',  # 2
            'policy_code',  # 1
            'out_prncp_inv',  # 2
            'out_prncp',  # 2
            'recoveries', # 2
            'title',  # 1
            'total_pymnt',  # 2
            'total_pymnt_inv',  # 2
            'total_rec_int',  # 2
            'total_rec_late_fee',  # 2
            'total_rec_prncp',  # 2
            'url',  # 1
         ]
         
loan.drop(remove, axis=1, inplace=True)
```

### 2. Assess assumptions on which statistical inference will be based

```javascript
if (isAwesome){
  return true
}
```

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
