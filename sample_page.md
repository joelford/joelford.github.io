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
