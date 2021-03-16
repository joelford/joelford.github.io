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

We are interested in whether a loan is fully paid or charged off. Lets limit ourselves to those two
```python
loan = loan.loc[loan['loan_status'].isin(['Fully Paid', 'Charged Off'])]
```

### 2. Exploration Class
The class Explore is used to automatically explore the data, do statistics and visualizations.

<details>
    <summary>Click to expand the Explore class</summary>
<br/><br/>
    
<details>
    <summary>Class initialization</summary>
    
```python
class Explore:    
    """ Explores relations with of a feature with a binary target
    Parameters
    -----------
    feat: string
        The feature being compared to the target
    binary_target: string,
        The binary target. Default is "loan_status"
    as_cat:
        To treat the feature as categorical data. Default False.  
    Attributes
    ----------
    dtype: string
        The feature's data type ('float' or 'object'), automatically determined
    target_values: list, length=2
        The values in the binary target, automatically calculated
    subset: Pandas dataframe
        A subset of the data with only the feature and binary target
    bucket: int, dict or None (default)
        If int, bucket dictionary is automatically made from user defined integer. 
        User defined dictionarys have form: {'bucket_range_name':[min,max],...}, eg. {'0':[0,0], '0+':[1,999]} (numerical)
            OR {'bucket_cat_name':[cat1,cat2,...],...} (categorical)
    """
    
    def __init__(self, 
                 feat, 
                 binary_target='loan_status', 
                 bucket=None,
                 as_cat=False):
        
        self.feat = feat
        self.binary_target = binary_target
        
        # Create a subset of the data from the feature and target
        self.subset = loan.loc[:,[self.feat, binary_target]]
        
        self.as_cat = as_cat
        
        # Drop NaNs, store the result and print for user
        self.num_missing = 0
        na_value_counts = self.subset[feat].isna().value_counts()
        
        if True in na_value_counts.index:
            orginal_num_rows = self.subset.shape[0]
            self.num_missing = na_value_counts[True]
            print(f'{feat} has {self.num_missing} ({(self.num_missing/orginal_num_rows)*100:.2f}%) missing rows. They are dropped.')
            self.subset.dropna(inplace=True)
                  
        else: 
            print(f'{feat} has no missing rows!')
        
        # Determine the feature data type
        if np.issubdtype(self.subset[self.feat], float) and not as_cat:
            self.dtype = 'float'
        else:
            self.dtype = 'object'
        
        # Initialize the self.bucket variable
        if bucket != None:
            if isinstance(bucket, int): # If int, automatically create buckets with user given value of quantiles
                self.bucket = self.quantile_bucketizer(bucket)
                
            elif isinstance(bucket, dict): # If user defined, use that
                self.bucket = bucket
            
            self.bucketize_data() # Finally, bucketize the data
            
        else:
            self.bucket = None
        
        # Create the target_values list, i.e. the two binary values in the target
        self.target_values = []
        self.target_values.append(self.subset[binary_target].unique()[0])
        self.target_values.append(self.subset[binary_target].unique()[1])
```            
</details>

<details><summary>Data transformation methods</summary>
<br/><br/>
    
<details><summary>Methods related to the bucket dictionary</summary>
The bucket dictionary allows for data to categorized based on "buckets". For example, we may want the data to be placed into buckets for all values 0 and everthing greater than 0.
<br/><br/>
<details><summary>bucket_key_words method</summary>
    
```python
    def bucket_key_words(self):
        '''Allows bucket dictionary to have key words inputted for convenience  
        '''
        if self.dtype == 'float' and not self.as_cat:
            # Create new bucket dict to replace old one (prevents dictionary iteration errors when replacing key:values)
            new_bucket = dict() 
            
            for key in self.bucket:
                value = self.bucket[key]
                
                # If a single integer passed, make the range [value, value]
                if isinstance(value, int):
                    new_bucket['['+str(value)+','+str(value)+']'] = [value, value]
                
                # If values are a list, search for key words (currently 'min' or 'max')
                elif isinstance(value, list):
                    new_list = [None for i in range(0, len(value))]
                    for i in range(0, len(value)):
                        new_list[i] = self.bucket[key][i]
                        
                        if isinstance(new_list[i], str):
                            if list_val == 'min':
                                new_list[i] = self.subset[self.feat].min()

                            elif list_val == 'max':
                                 new_list[i] = self.subset[self.feat].max()
                        
                    new_bucket[key] = new_list

                # If value is a string
                elif isinstance(value, str):
                    # '+' in the string value indicates we want a range greater (or equal) to the value
                        # Input should be of form '{int_val}+' OR '{int_val}+=' OR '{int_val}=+'
                    if '+' in value:
                        
                        # = present, so we want to replace with closed interval '[int_val,max]':[int_val,max]
                        if '=' in value:
                            int_val = int(value[:-2]) #
                            left_bracket = '['
                            
                        # No =, so we want to replace with left open interval '(int_val,max]':[int_val,max] 
                        else: 
                            int_val = int(value[:-1])
                            left_bracket = '('
                        
                        max_val = self.subset[self.feat].max()
                        
                        # Set new key with proper formatting and value, delete old key
                        new_bucket[left_bracket+str(int_val)+','+str(max_val)+']'] = [int_val, max_val]
                    
                    else:
                        # Determine if a string number (ex: '123') is inputted, range to be [int_value, int_value]
                        is_number = True
                        for s in value:
                            if not s.isdigit():
                                is_number = False
                        
                        if is_number:
                            int_value = int(value)
                            new_bucket['['+value+','+value+']'] = [int_value, int_value]
                         
            # Replace with new bucket
            self.bucket = new_bucket
```

</details>

<details><summary>quantile_bucketizer method method</summary>
    
```python    
    def quantile_bucketizer(self, numq=4, frmt='.2f'):
        ''' Automatically creates a bucket dictionary using quantiles for the data
        Parameters
        -----------
        numq: int
            Number of quantiles to break the data into
        frmt: string
            Formatting for the values, '.2f' for two decimals, 'nodec' for no decimals
        '''
        
        quantiles = dict()
        
        for i in range(-1,numq): #-1 to include the "zeroeth quartile" (min)
            val = self.subset[self.feat].quantile(( 1/numq) * (i+1))
            quantiles[i+1] = val

        bucket = dict()
        
        for key in quantiles:
            if key < numq:
                lower = quantiles[key]
                upper = quantiles[key + 1]

                if frmt == '.2f':
                    lower_format = f'{lower:.2f}'
                    upper_format = f'{upper:.2f}'

                elif frmt == 'nodec':
                    lower_format = f'{lower:.0f}'
                    upper_format = f'{upper:.0f}'

                if key != numq-1: 
                    key_string = '['+lower_format+','+upper_format+')'
                
                else: # If it is the last value, we want fully closed brackets
                    key_string = '['+lower_format+','+upper_format+']'
                
                bucket[key_string] = [lower, upper]

        print(f'Bucket automatically computed for {numq} quantiles:\n{bucket}')
        return bucket
```    

</details>  

<details><summary>bracket_type method</summary>
    
```python    
    def bracket_type(self, value):
        ''' Uses simple regular expressions to determine bracket type (open, left open, right open, closed)
        from the key in the bucket dictionary. 
        Parameters| value, string
        '''
        if re.match('(\[.*\])', value) != None: return 'c'
        
        elif re.match('(\(.*\])', value) != None: return 'l'
        
        elif re.match('(\[.*\))', value) != None: return 'r'
        
        elif re.match('(\(.*\))', value) != None: return 'o'
        
        else: return False
```
 
</details>
    
<details><summary>bucketize_data method</summary>

```python
    def bucketize_data(self, new=None):
        '''Using the bucket dictionary, sort the data into their buckets
        Parameters| new: User defined bucket dictionary or None (default) 
        '''
        if new == None and 'bucketized_'+self.feat in self.subset.columns:
            pass # Do nothing if already bucketized
        
        else:
            if self.bucket == None and new == None: 
                # Create a bucket dictionary if necessary, default 10
                self.bucket = self.quantile_bucketizer(10)
                
            if new != None:
                if isinstance(new, int): # If int given, automake with quantile_bucketizer function
                    self.bucket = self.quantile_bucketizer(numq=new)
                
                else:    
                    self.bucket = new # If new bucket given (and not int), set it
                
            # In case there are bucket key words
            self.bucket_key_words()
            
            # With everything set up, perform the bucketization
            self.bucketize()
```

</details>

<details><summary>bucketize method</summary>

```python
    def bucketize(self):
        ''' Bucketizes the data. self.bucket_dict format should have a format like this when called:
        {'[num1,num2]':[num1,num2], 
         '[num3,num4)':[num3,num4],
         '(num5,num6]':[num5,num6],
         '(num7,num8)':[num7,num8]}
         The bracket types in the string key determines the type of inclusion used.
         Calling self.bucket_key_words() prior ensures any valid key words are put into this format
        '''
        # Categorical bucketizing
        if self.dtype == 'object' or self.as_cat: 
            for key in self.bucket:
                indices = (self.subset[self.feat].isin(self.bucket[key]))
                self.subset.loc[indices, 'bucketized_'+self.feat] = key
                
                
        # Numerical bucketizing
        else:
            for key in self.bucket:
                # Determine the type of inclusion
                bracket_type = self.bracket_type(key)

                # Extract the boundaries 
                lower = self.bucket[key][0]
                upper = self.bucket[key][1]

                # Determine the indices using the correct inclusion
                if bracket_type == 'c': # Closed
                    indices = (self.subset[self.feat] >= lower) & (self.subset[self.feat] <= upper)

                elif bracket_type == 'l': # Left open
                    indices = (self.subset[self.feat] > lower) & (self.subset[self.feat] <= upper)

                elif bracket_type == 'r': # Right open
                    indices = (self.subset[self.feat] >= lower) & (self.subset[self.feat] < upper)

                elif bracket_type == 'o': # Open
                    indices = (self.subset[self.feat] > lower) & (self.subset[self.feat] < upper)

                # Using the indices, replace the data with the bucket key in a new 'bucketized'+self.feat column
                self.subset.loc[indices, 'bucketized_'+self.feat] = key
```

</details>  

</details>                

<details><summary>Other Transformations method</summary>
<br/><br/>
    
<details><summary>log_tansform method</summary>

```python
    def log_transform(self):
        ''' Performs a log transformation of the data, only if not done already
        '''
        if 'log_'+self.feat in self.subset.columns:
            pass # Do nothing if already log transformed
        
        else: 
            self.subset['log_'+self.feat] = self.subset[self.feat].apply(lambda x: np.log(x+1))
```    

</details>

<details><summary>delete_values method</summary>

```python
    def delete_values(self, values):
        '''Deletes values from a given list. List values can be strings, integers or length 2 lists (ranges to delete)
        '''
        for val in values:
            if isinstance(val, str) or isinstance(val, int):
                self.subset = self.subset[self.subset[self.feat] != val]
            
            elif isinstance(val, list):
                self.subset = self.subset[~self.subset[self.feat].between(val[0], val[1], inclusive=True)]
 ```
 
</details>
 
</details>

</details>
 
<details><summary>Main Methods method</summary>
Main methods for statistics and visualization.
<br/><br/>    
    
<details><summary>do_stats method</summary>

```python
    def do_stats(self, prefix=''):
        ''' Runs statistics on the data.
        Parameters| prefix: string, for using transformed data
        '''
        feat = self.add_prefix(prefix)
        
        # Statistics for numerical comparison, using Mann-Whitney U test
        if self.dtype == 'float' and not self.as_cat and not 'bucketized' in feat:
            grouped_by_values = {key:None for key in 
                                         self.subset[self.binary_target].unique()}

            for i in self.subset[self.binary_target].unique():
                grouped_by_values[i] = self.subset[
                    self.subset[self.binary_target] == i][feat]

            median_diff = grouped_by_values[self.target_values[0]].median() \
                - grouped_by_values[self.target_values[1]].median()
            
            self.stats = spstats.mannwhitneyu(grouped_by_values[self.target_values[0]], 
                                   grouped_by_values[self.target_values[1]])
            

            print(self.stats)
            print(f'{feat} median difference for {self.target_values[0]} - {self.target_values[1]}: {median_diff:.4f}')
            
        # For categorical-to-categorical comparison (ChiSq independence on contingency table)
        else:
            contingency_table = pd.crosstab(self.subset[self.binary_target], 
                                            self.subset[self.feat])
            
            s = spstats.chi2_contingency(contingency_table)
            self.stats = {'ChiSq':s[0], 'p_value':s[1]}

            print(self.stats)     
```   

</details>

<details><summary>plot_counts method</summary>

```python
    def plot_counts(self, prefix=''):
        ''' For plotting a graph of the data counts, to see the distribution
        Parameters| prefix: string, for using transformed data
        '''
        feat = self.add_prefix(prefix)
        fig, ax = plt.subplots(figsize=(12,4))
        
        # For numerical data, use a histogram (don't do if treated as categorical or prefix is bucketized)
        if self.dtype == 'float' and not self.as_cat and not 'bucketized' in feat:
            ax = sns.histplot(self.subset[feat], ax=ax)
            ax.set_title(feat+' Counts')
        
        # For categorical data, use a countplot
        else: 
            ax = sns.countplot(x=self.subset[feat], 
                                  order=sorted(self.subset[feat].unique(), key=self.natural_key),
                                  ax=ax,
                                  color='orange')
            plt.setp(ax.get_xticklabels(), rotation=45)

            ax.set_title(feat+' Counts')    
            ax.set_xlabel(feat)
        
        plt.tight_layout()
```

</details>

<details><summary>plot_num method</summary>

```python
    def plot_num(self, prefix=''):
        ''' For plotting a comparison between the binary target_values for numerical data
        Parameters| prefix: string, for using transformed data
        '''
        feat = self.add_prefix(prefix)
        fig, ax = plt.subplots(figsize=(12,4))
        
        ax = sns.boxplot(x=feat, y=self.binary_target, 
                          data=self.subset, ax=ax)
        ax.set_title(f'Boxplots comparing {self.target_values[0]} vs {self.target_values[1]} for '+feat)
        
        plt.tight_layout()
```
</details>

<details><summary>plot categorical data method</summary>

```python
    def plot_cat(self, prefix='', target_value_compared=1):
        ''' For comparing categorical data, using a barplot to compare rates of target_value_compared
        for each category.
        Parameters
        -----------
        prefix: string, 
            For using transformed data
        target_value_compared: int, 
            The index in the target_values list of the target value to be compared. Default 1.
        '''
        feat = self.add_prefix(prefix)
        fig, ax = plt.subplots(figsize=(12,4))
        
        rate_name = self.target_values[target_value_compared]
        
        rate = self.subset.groupby(
            feat)[self.binary_target].value_counts(normalize=True).loc[:, rate_name]

        pal = sns.color_palette("Reds_d", len(rate.values))
        rank = rate.values.argsort().argsort()
        ax = sns.barplot(x=rate.index, 
                        y=rate.values,
                        order=sorted(rate.index, key=self.natural_key),
                        palette=np.array(pal)[rank])
        plt.setp(ax.get_xticklabels(), rotation=45)

        ax.set_title(f'{rate_name} Rate vs ' + feat)
        ax.set_ylabel(f'Rate {rate_name}')
        ax.set_xlabel(feat)
        
        plt.tight_layout()
```        

</details>

</details>

<details><summary>Other methods</summary>
Appendage methods for working with the data or convenience

<details><summary>add_prefix method</summary>

```python
    def add_prefix(self, prefix):
        '''Takes a given prefix (currently only looking at first letter) 
        and returns appriopriate feature name (changing nothing if empty string '' passed)
        Also calls for the appropriate transformation (only actually done if necessary, allows for transformations
        to be performed without explicitly calling for it). 
        Returns the prefix + feature name. '''

        p = ''

        if len(prefix) > 0:
            if prefix[0] == 'l':
                self.log_transform()
                p = 'log_'

            elif prefix[0] == 'b':
                self.bucketize_data()
                p = 'bucketized_'

        return p + self.feat
```

</details>

<details><summary>natural_key methods</summary>

```python
    def natural_key(self, string_):
        ''' To "naturally" sort strings with numbers (ex: so [2,9] comes before [10,99])
        Only used in plotting count & categorical graphs with string number categories
        I strip the first/last (string_[1:-1]) as this will look at things of the form '[num1,num2]'
        See https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/'''
        string_ = str(string_)
        
        # If brackets are present, strip those so the first number is what's taken into account
        if isinstance(string_, str) and ('[' in string_ or '(' in string_):
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_[1:-1])]
        
        else:
            return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
```

</details>

<details><summary>auto convenience method</summary>

```python
    def auto(self, log=False, bucket=None):
        ''' Convenience method for automatiically doing stats and the desired plots
        '''
        self.do_stats()
        only_buckets = False
        
        bucket_px = ''
        if bucket != None:
            if not isinstance(bucket, bool) and isinstance(bucket, int):
                self.bucketize_data(bucket)
            
            if bucket == 'only buckets':
                only_buckets = True
            
            bucket_px = 'bucket'
        
        log_px = ''
        
        if log: 
            log_px = 'log'
               
        if only_buckets:
            self.plot_counts(bucket_px)
            self.plot_cat(bucket_px)
        
        elif self.dtype == 'float':
            self.plot_counts(log_px)
            self.plot_num(log_px)
            self.plot_cat('bucket')
        
        elif self.dtype == 'object':
            self.plot_counts(log_px)
            self.plot_cat(bucket_px)
```

</details>

</details>

</details>

### 3. Support the selection of appropriate statistical tools and techniques

<img src="images/dummy_thumbnail.jpg?raw=true"/>

### 4. Provide a basis for further data collection through surveys or experiments

Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. 

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).
