# KNN_imputation
Imputing missing value using KNN.
## Install
pandas, numpy are required

```
git clone 
# install dependencies
pip install -r requirements.txt
```
## Usage
```
from knn_imputation import KnnImpte
KNNImp = KnnImpte(k, thershold)
```
The imputation of missing values in the training set.
(```target_attr``` is default a string, ```fillin_attr``` is default a list of string, ```df``` is the traning dataframe)
```
'''
target_attr: the attribute to be imputed
fillin_attr: the atrributes to determine the distance
'''
df[target_attr] = KNNImp.fit_transform(target_attr, fillin_attr, df)
```
The imputation of missing value in the testing set. Using the instances in training set to impute the testing set.
```
# testing set
df_ts[target_attr] = transfrom(df)
```

