# Moscow Real Estate Model

## Group members
- Keri Chen
- Arth Shukla
- WonJae Lee
- Ashley Chu
- Cynthia Delira

# Abstract 
Real estate is the foundation for many life milestones like owning a home, starting a family or a business, and more. However, it may be hard to break into the real estate world without first doing a lot of research and planning because real estate is after all, an investment. By building a ML model that helps predict the price of home, this can hopefully help to make the process easier for prospective homeowners and sellers. The data we will be using encompasses more than 5 million Russian real estate sales from 2018 - 2021 and has multiple variables such as the price of a house, listing date and time, geographic location, region, and information about the building (type, storeys, floor, living rooms, rooms). Although our dataset is in the Russian market, it provides us a lot of data points that can allow us to learn more about the different models and generalize it to different markets.

We will perform some EDA analysis to view the correlation of the different factors, and then build a linear regression model using CART regression, logistic regression, and random forest. We will then evaluate the performance of our model using mean absolute error (MAE).

# Background

The real estate market has been a pivotal factor and contributor in the economy as, according to the National Association of Home Builders, housing’s combined contribution to gross domestic product (GDP) generally averages 15-18%<a name="nahb"></a>[<sup>[1]</sup>](#nahb). This percentage is calculated based on both residential investment and consumption spending on housing. Not only is housing a contributor to the economy but it is also an important asset to people’s lives as it not only signifies having a place to sleep in but is often perceived as a way to show one’s social status and a valuable asset where money can be allocated.


Despite the importance and high contribution that the market is to the economy, it has many factors that can quickly change the market. Although different factors can influence the real estate market, one of the most important factors is demographics<a name="keyfactors"></a>[<sup>[2]</sup>](#keyfactors).

Demographics consists of the data of the population near the location which affects the pricing of the house and also the demand the property has. Places in and near a major city could be more expensive due the proportion of square footage and price<a name="demographics">,</a>[<sup>[3]</sup>](#demographics) since major cities usually have limited land to be developed or already has no more space for new developments. Alhtough real estate predictions across different areas (urban, rural, suburban) can vary due to differences in land use, when considering a single location (e.g. one city) where land use, housing supply, etc., are more similar, demographics prevail as the key difference when comparing real estate.

Our exploration will, therefore, focus on Moscow, the capital of Russia which has a quickly-growing real estate market. Because the market is developing, it can be difficult for an average person to determine which variables contribute most to real estate pricing. By building a model to predict real estate pricing (in rubles), we aim to make distilling this demographic information easier on a larger scale.

There has been a lot of prior (and ongoing) research within the real estate industry, especially real estate companies such as Zillow with their “Neural Zestimate<a name="Zestimate"></a>[<sup>[4]</sup>](#Zestimate)," Redfin with their “Redfin Estimate<a name="Redfin"></a>[<sup>[5]</sup>](#Redfin)," and many other real estate companies also have their own models for estimating home prices. Since each model is built differently, this leads to varying price estimations. However, the bases of the models are similar as they take in large amounts of previous transactions and/or MLS data to get various variables to find good features to base the model off of as they keep retraining to get better results.

# Problem Statement

The real estate market can be a turbulent and rapidly changing environment, where it is often hard to predict the actual value due to many factors.Due to the multitude of different constants, we will focus our model on the general description of the property. We aim to make it easier for people to get this type of information by training a ML model on a large dataset of previous home purchases in order to predict what price point a home may be at. 

# Data

Our current best candidate is the following dataset of <a href="https://www.kaggle.com/datasets/mrdaniilak/russia-real-estate-20182021">Russian Real Estate pricing from 2018-2021.</a> The dataset contains an incredible 5 million+ data points, with no null values and only a few thousand duplicate rows. Therefore, our data is very-well poised to avoid generalization without uses of techniques like cross-validation. 

This massive dataset means training could takes many days or even weeks given our computational resources, which is not feasible. Since demographics data can vary between cities/counties. However, our exploration will primarily focus on Moscow. Thus, we are able to limit the size of our data to about 1/10th of the original dataset. Furthermore, if computational cost continues to be an issue, we may randomly sample a subset of our data to train (this will not harm any assumptions for the regression models we will use, since it does not violate any assumptions about the data which these models require)

There are 13 variables, 2 categorical, 2 location-based, and the rest ordinal. We will be removing the latitude and longitude columns as these prevent ethical issues regarding the location of homeowners and intense violations of privacy.

Each observation contains the price of a house, listing date and time, geographic location, region, and information about the building (type, storeys, floor, living rooms, rooms). Notably, it does not contain square footage, which is a landmark in much of the American real estate market.

Critical variables mostly encompass the house descriptions and the time of publishing. We will need to one-hot encode building type. Building type will not largely increase the width of the design matrix.

Finally, we will need to convert data and time of publication to only the year, and potentially also the month, in case we’d like to do time series analysis. As mentioned earlier, we’ll also remove the latitude and longitude due to concerns of privacy. Finally, for our non-tree models, we will also normalize our data points by z-score, since data like price in rubles will be orders of magnitude larger than number of rooms.

# Proposed Solution

Note that we discuss error metrics, including justifications for L1 loss (MAE), in the Evaluation Metrics section.

Before discussing our implementation, regarding benchmark models: there are some models available on Kaggle using time series analysis, which might result in good outputs. However, there are no significant authorities on Russian real estate pricing in machine learning, especially since this is an emerging market. Furthermore, American authorities on real estate prediction often keep their models internal as a part of their business model, so it is difficult to use existing robust benchmark models without APIs and the sort.
First, it is important to note that our dataset is massive. With over 5 million samples, our model will certainly generalize well, but this also means we may have too many confounding variables and our model may not reach high enough MAE. During EDA, we will determine cities which contain interesting data, and we can fragment our data by city. Depending on computational resources and time constraints, we may choose multiple cities, or only use one.

Second, regardless of which or how many cities we use, this data is simply far too massive for any form of CV. Additionally, CV is not necessary here, since our validation set is likely to generalize well.
Finally, luckily much of the data is ordinal, with few categorical variables with limited possible values. For our regression models, we may try to avoid extra data points such that we don’t have too many features in our design matrix. However, to attempt to include these features in at least one model, we will also try random forests.

- CART Regression
- Linear regression using L1 loss
- After performing EDA, if certain metrics seem like they could use polynomial features, we can also try polynomial regression using L1 loss.
- Random Forests to include categorical variables.


We can also try variants of linear and polynomial regression using L2 regularization. It is unlikely that many of these features will be confounding (though we can confirm with EDA), so L2 regularization is likely more reasonable. We can also try mixed regularization in case some features are, indeed, confounding.

Then, if we have enough computational resources, we can perform grid search on different hyperparameters for model selection. However, if this is not feasible, we can empirically justify pruning techniques, regularization mix, etc.

Finally, we will use sklearn for all implementations for 1) readable code, and 2) efficient, thoroughly tested implementations of the algorithms discussed above. While tools like Keras do have gpu acceleration, these methods aren’t as useful for our models as compared to neural network models.


# Evaluation Metrics

The three most common metrics for regression are mean squared error (MSE), mean absolute error (MAE) and root mean squared error (RMSE). MSE and RMSE heavily penalize outliers, while MAE proportionately penalizes all errors. Our data includes some more extreme outliers (10 living rooms, 39th floor, etc). For these ‘extreme’ sorts of houses, there are also many extra possible factors beyond measurable features like number of rooms; for example, the ‘art’ of designing expensive homes with luxury features. So, using MSE or RMSE would likely bias our model to these extreme outliers while lowering our model’s success in gauging prices for a majority of houses on the market. Conversely, MAE would result in a better representation of the data for a majority of ‘normal’ cases. Therefore, we will stick to MAE.

# Results


# Real Estate Analysis

## Setup and Load Data

First, we retrieve the dataste from https://www.kaggle.com/datasets/mrdaniilak/russia-real-estate-20182021.

If using the below cell, make sure you have a Kaggle API token in a `kaggle.json` file in `~/.kaggle/`. Otherwise, please download the data manually and place it under a folder `./data/`.


```python
# !mkdir data
# !kaggle datasets download mrdaniilak/russia-real-estate-20182021
# !mv ./russia-real-estate-20182021.zip ./data/russia-real-estate-20182021.zip
```


```python
# import zipfile
# with zipfile.ZipFile('./data/russia-real-estate-20182021.zip', 'r') as zip_ref:
#     zip_ref.extractall('./data')
```

Now, we may proceed with exploration.


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```


```python
DATA_CSV_PATH = './data/all_v2.csv'

# load csv data
df = pd.read_csv(DATA_CSV_PATH)

# remove duplicate data
df = df.drop_duplicates()
```


```python
df.sample(5)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>date</th>
      <th>time</th>
      <th>geo_lat</th>
      <th>geo_lon</th>
      <th>region</th>
      <th>building_type</th>
      <th>level</th>
      <th>levels</th>
      <th>rooms</th>
      <th>area</th>
      <th>kitchen_area</th>
      <th>object_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3062183</th>
      <td>2530000</td>
      <td>2019-12-20</td>
      <td>01:30:18</td>
      <td>50.601726</td>
      <td>36.601702</td>
      <td>5952</td>
      <td>1</td>
      <td>2</td>
      <td>9</td>
      <td>1</td>
      <td>35.00</td>
      <td>10.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41186</th>
      <td>1190000</td>
      <td>2018-09-14</td>
      <td>02:12:39</td>
      <td>54.947141</td>
      <td>82.958596</td>
      <td>9654</td>
      <td>1</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>41.36</td>
      <td>11.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2835037</th>
      <td>3100000</td>
      <td>2019-11-05</td>
      <td>06:27:25</td>
      <td>57.115194</td>
      <td>65.559313</td>
      <td>3991</td>
      <td>2</td>
      <td>1</td>
      <td>14</td>
      <td>1</td>
      <td>39.00</td>
      <td>12.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46232</th>
      <td>1990000</td>
      <td>2018-09-14</td>
      <td>15:21:55</td>
      <td>45.097870</td>
      <td>39.004688</td>
      <td>2843</td>
      <td>0</td>
      <td>5</td>
      <td>16</td>
      <td>1</td>
      <td>40.00</td>
      <td>16.0</td>
      <td>11</td>
    </tr>
    <tr>
      <th>1712186</th>
      <td>2400000</td>
      <td>2019-04-30</td>
      <td>11:00:17</td>
      <td>55.188396</td>
      <td>61.329709</td>
      <td>5282</td>
      <td>1</td>
      <td>3</td>
      <td>9</td>
      <td>3</td>
      <td>55.00</td>
      <td>9.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


Note that this Kaggle Dataset was originally pulled from GeoNames (http://www.geonames.org/), which has its own "regions" separated by county. Our city of interest, Moscow, has ID 81.

Addtionally, we will give categorical data appropriate labels given by the dataset spec.

Finally, while the dataset used Russian rubles for real estate price, we use US Dollars for more interpretable loss (since the Ruble prices are in the millions, and since we are more familiar with the US Dollar).


```python
MOSCOW_CODE = 81
MAX_NUM_SAMPLES = 100000
SEED = 69

RUBLE_TO_DOLLAR = 0.012

moscow_df = df.loc[df['region'] == MOSCOW_CODE]
moscow_df = moscow_df.drop(['time', 'geo_lat', 'geo_lon', 'region'], axis=1)
moscow_df['date'] = moscow_df['date'].apply(lambda x: int(x[:4]))

moscow_df['object_type'] = moscow_df['object_type'].replace(1, 'preowned').replace(11, 'new')
moscow_df['building_type'] = moscow_df['building_type'].replace(0, 'other').replace(1, 'panel').replace(2, 'monolithic').replace(3, 'brick').replace(4, 'blocky').replace(5, 'wooden')

# -1 means studio apartment, so we replace with 0 (since studio apartments have no extra rooms)
# there are not other datapoints with value 0
moscow_df['rooms'] = moscow_df['rooms'].replace(-1, 0)

# remove rows with errorneous data
moscow_df = moscow_df[moscow_df['price'] >= 0]
moscow_df = moscow_df[moscow_df['rooms'] >= 0]

#  convert to US dollar per conversion rate as of June 12, 2023
moscow_df['price'] = moscow_df['price'] * RUBLE_TO_DOLLAR

# cap number of elements
moscow_df = moscow_df.sample(MAX_NUM_SAMPLES, random_state=SEED)
```


```python
moscow_df.isnull().values.any()
```


    False



```python
moscow_df.sample(5)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>date</th>
      <th>building_type</th>
      <th>level</th>
      <th>levels</th>
      <th>rooms</th>
      <th>area</th>
      <th>kitchen_area</th>
      <th>object_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2263684</th>
      <td>62400.00</td>
      <td>2019</td>
      <td>monolithic</td>
      <td>20</td>
      <td>24</td>
      <td>2</td>
      <td>62.0</td>
      <td>12.0</td>
      <td>preowned</td>
    </tr>
    <tr>
      <th>4742429</th>
      <td>130200.00</td>
      <td>2020</td>
      <td>monolithic</td>
      <td>23</td>
      <td>25</td>
      <td>3</td>
      <td>92.3</td>
      <td>27.0</td>
      <td>preowned</td>
    </tr>
    <tr>
      <th>844302</th>
      <td>36751.32</td>
      <td>2018</td>
      <td>panel</td>
      <td>23</td>
      <td>25</td>
      <td>1</td>
      <td>19.9</td>
      <td>5.0</td>
      <td>new</td>
    </tr>
    <tr>
      <th>3987839</th>
      <td>51600.00</td>
      <td>2020</td>
      <td>panel</td>
      <td>10</td>
      <td>10</td>
      <td>1</td>
      <td>42.8</td>
      <td>10.5</td>
      <td>preowned</td>
    </tr>
    <tr>
      <th>1315428</th>
      <td>87193.08</td>
      <td>2019</td>
      <td>monolithic</td>
      <td>11</td>
      <td>24</td>
      <td>3</td>
      <td>80.6</td>
      <td>13.9</td>
      <td>new</td>
    </tr>
  </tbody>
</table>
</div>


## Exploration

First, we'll look at the data distributions for the ordinal data.


```python
moscow_df.describe()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>price</th>
      <th>date</th>
      <th>level</th>
      <th>levels</th>
      <th>rooms</th>
      <th>area</th>
      <th>kitchen_area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.000000e+05</td>
      <td>100000.000000</td>
      <td>100000.000000</td>
      <td>100000.000000</td>
      <td>100000.000000</td>
      <td>100000.000000</td>
      <td>100000.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.707663e+04</td>
      <td>2019.310120</td>
      <td>7.053700</td>
      <td>12.752370</td>
      <td>1.776320</td>
      <td>51.592710</td>
      <td>10.362995</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.556933e+04</td>
      <td>0.880621</td>
      <td>5.692804</td>
      <td>7.411351</td>
      <td>0.862656</td>
      <td>21.748209</td>
      <td>6.834285</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.800000e+01</td>
      <td>2018.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>2.400000</td>
      <td>0.120000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.600000e+04</td>
      <td>2019.000000</td>
      <td>3.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>38.000000</td>
      <td>7.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.160000e+04</td>
      <td>2019.000000</td>
      <td>5.000000</td>
      <td>12.000000</td>
      <td>2.000000</td>
      <td>46.000000</td>
      <td>9.700000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.140000e+04</td>
      <td>2020.000000</td>
      <td>10.000000</td>
      <td>17.000000</td>
      <td>2.000000</td>
      <td>61.900000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.572896e+06</td>
      <td>2021.000000</td>
      <td>36.000000</td>
      <td>38.000000</td>
      <td>9.000000</td>
      <td>997.000000</td>
      <td>1131.000000</td>
    </tr>
  </tbody>
</table>
</div>


Each column is skewed right, meaning we have some extreme outliers for each column. This is because in the real estate market, while most "normal" places have a similar price, the price ceiling for real estate can be very high. While these outliers are sparse, they could still bias our model.

Next, we can plot each variable against price to look for possible correlations. We will only looks at data points with price less than $2 \cdot 10^7$ to get better plots by removing price outliers. We will also plot regression lines for each to quantify per-variable correlation strength.


```python
from scipy.stats import linregress

def calc_R2(x, y, ax=None, **kwargs):
    ax = ax or plt.gca()
    _, _, r_value, _, _ = linregress(x=x, y=y)
    ax.annotate(f'$R^2 = {r_value ** 2:.2f}$', xy=(.05, 1), xycoords=ax.transAxes, fontsize=8, ha='left', va='top')

g = sns.pairplot(moscow_df.loc[moscow_df['price'] < 2e7].sample(1000, random_state=21), kind='reg', y_vars=['price'], plot_kws={'line_kws':{'color':'red'}})

g.map_upper(calc_R2)
plt.show()
```


    
![png](./img/output_26_0.png)
    


Data and level have near-zero correlations. The number of levels in the building (i.e. building size) and number of rooms, as well have kitchen area, seem like they might have some significance. The most important (single) variable seems to be area. However, even here we don't have a strong correlation. Hopefully combining these variables into a multivariate regression will lead to stronger correlation.

Additionally, all of the correlations seem to be closest to linear (as opposed to some polynomial fit). So, a polynomial regression may not perform better than a linear regression.

However, in the above we only use about 1000 samples (for efficiency). We can see these correlation results across the dataset more easily with a heatmap:


```python
sns.heatmap(moscow_df.drop(['building_type', 'object_type'], axis=1).corr(), annot=True)
plt.show()
```


    
![png](./img/output_28_0.png)
    


Notably, `date` and `level` have a stronger correlation than from our 1000 samples, but each individual variable still does not have a strong enough correlation for prediction.

Thus, we proceed to fitting some models.

## Preprocessing Data for Models


```python
from sklearn.model_selection import train_test_split

X = moscow_df[['date', 'building_type', 'level', 'levels', 'rooms', 'area', 'kitchen_area', 'object_type']]
y = moscow_df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

print(f'{len(y_train)} train samples; {len(y_test)} test samples')
```

    70000 train samples; 30000 test samples


Now, we will one-hot the categorical data using sklearn's one hot encoder.


```python
from sklearn.preprocessing import OneHotEncoder

for col in ['building_type', 'object_type']:

    one_hot = OneHotEncoder()
    one_hot.fit(X_train[[col]])

    X_train.loc[:, one_hot.categories_[0]] = one_hot.transform(X_train[[col]]).todense()
    X_test.loc[:, one_hot.categories_[0]] = one_hot.transform(X_test[[col]]).todense()

    X_train = X_train.drop(col, axis=1)
    X_test = X_test.drop(col, axis=1)

X_train.sample(5)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>level</th>
      <th>levels</th>
      <th>rooms</th>
      <th>area</th>
      <th>kitchen_area</th>
      <th>blocky</th>
      <th>brick</th>
      <th>monolithic</th>
      <th>other</th>
      <th>panel</th>
      <th>wooden</th>
      <th>new</th>
      <th>preowned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4065819</th>
      <td>2020</td>
      <td>9</td>
      <td>25</td>
      <td>1</td>
      <td>35.7</td>
      <td>17.00</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1038879</th>
      <td>2019</td>
      <td>4</td>
      <td>10</td>
      <td>1</td>
      <td>41.8</td>
      <td>11.50</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4628192</th>
      <td>2020</td>
      <td>2</td>
      <td>25</td>
      <td>1</td>
      <td>39.5</td>
      <td>18.10</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>467894</th>
      <td>2018</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>45.0</td>
      <td>7.00</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4095709</th>
      <td>2020</td>
      <td>15</td>
      <td>21</td>
      <td>2</td>
      <td>60.6</td>
      <td>12.94</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


Next, we will scale the data. Note that not all models require scaled data; while models like linear regression require scaling to avoid overemphasis of certain datapoints, models like descision trees (and, by extension, random forests) are not affected by unscaled variables (though, of course, scaled data won't negatively impact performance models like descision trees, either).


```python
from sklearn.preprocessing import StandardScaler

ordinal_cols = ['date', 'level', 'levels', 'rooms', 'area', 'kitchen_area']
X_train_ordinal, X_test_ordinal = X_train[ordinal_cols], X_test[ordinal_cols]

scaler = StandardScaler()
scaler.fit(X_train_ordinal)

X_train_scaled, X_test_scaled = X_train, X_test

X_train_scaled.loc[:, scaler.feature_names_in_] = scaler.transform(X_train_ordinal)
X_test_scaled.loc[:, scaler.feature_names_in_] = scaler.transform(X_test_ordinal)

X_train_scaled.sample(5)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>level</th>
      <th>levels</th>
      <th>rooms</th>
      <th>area</th>
      <th>kitchen_area</th>
      <th>blocky</th>
      <th>brick</th>
      <th>monolithic</th>
      <th>other</th>
      <th>panel</th>
      <th>wooden</th>
      <th>new</th>
      <th>preowned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>531405</th>
      <td>-1.486208</td>
      <td>-0.364642</td>
      <td>-1.052022</td>
      <td>-0.900040</td>
      <td>-1.443469</td>
      <td>-0.713346</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>852950</th>
      <td>-1.486208</td>
      <td>0.514697</td>
      <td>0.432760</td>
      <td>0.257887</td>
      <td>0.166966</td>
      <td>1.310532</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5106064</th>
      <td>1.919550</td>
      <td>-0.364642</td>
      <td>-0.917041</td>
      <td>-0.900040</td>
      <td>-0.302934</td>
      <td>0.212611</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3020574</th>
      <td>-0.350955</td>
      <td>-0.188774</td>
      <td>1.647581</td>
      <td>0.257887</td>
      <td>0.399636</td>
      <td>0.278751</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2738752</th>
      <td>-0.350955</td>
      <td>-0.012906</td>
      <td>0.567740</td>
      <td>-0.900040</td>
      <td>-0.348556</td>
      <td>-0.184227</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>


## Model Fitting

When we began training models, an immediate issue presented itself: we didn't have sufficient compuational resources to handle very large data using only Sklearn's cpu-only implenetaions. This made training very slow and hyperparameter tuning out of the question.

However, training only on our data with hyperparams chosen heurisitically did not offer good enough results. So, in order to allow for hyperparam tuning on our large data, we implement Nvidia's [RAPIDS](https://rapids.ai/) API, which offers models with similar syntax to Sklearn, but with GPU acceleration. In particular, RAPIDS includes the CuML package, which accelerates training significantly.

The main challenge with using RAPIDS was technical implementation: we acheived most stability on WSL2 Ubuntu on Windows with CUDA 11.5, and some functions in CuML seem less stable (i.e. more prone to crashing or hanging on computation) with this setup.

**Finally**, although we will be testing different models, we will not be peforming Nested CV for algorithm selection. Even with the RAPIDS API, performing Nested CV would be simply too computationally intensive, and also likely wouldn't realize better results on our large data.

### Rapids CuML Setup

First, we convert our data using CuDF and CuPY to allow our CuML models to use GPU acceleration.


```python
import cudf
import cupy as cp

def to_cudf(pd_df):
    data = dict()

    for col in pd_df.columns:
        data[col] = pd_df[col].to_numpy(dtype=np.float32)

    return cudf.DataFrame(data)

def to_cupy(pd_df):
    return cp.from_dlpack(pd_df.to_dlpack())
```


```python
X_train_cudf = to_cudf(X_train)
X_test_cudf = to_cudf(X_test)

X_train_cupy = to_cupy(X_train_cudf)
X_test_cupy = to_cupy(X_test_cudf)
```


```python
X_train_scaled_cudf = to_cudf(X_train_scaled)
X_test_scaled_cudf = to_cudf(X_test_scaled)

X_train_scaled_cupy = to_cupy(X_train_scaled_cudf)
X_test_scaled_cupy = to_cupy(X_test_scaled_cudf)
```


```python
y_train_cudf = cudf.Series(y_train.to_numpy())
y_test_cudf = cudf.Series(y_test.to_numpy())

y_train_cupy = to_cupy(y_train_cudf)
y_test_cupy = to_cupy(y_test_cudf)
```

Next, we found that Sklearn's `GridSearchCV` and `RandomizedSearchCV` were somewhat unstable with RAPIDS on our machines; in particular, there seemed to be some issues regarding the way Sklearn initializes new models when performing hyperparam tuning.

For this reason, we create a `custom_grid_search` function which, though containing fewer features, is more stable with RAPIDS.

Addtionally, our cross-validation will use 3 folds. Because our data is larger, fewer splits in our CV will still give our models enough data to train while reducing variability in predicitons.

While this function allows us to perform CV using RAPIDS in a more stable manner, it has the drawback of not being able to view variability of fit curves. However, given our large data and low `k=3`, our varaiblity in prediction from CV will likely not be an issue.


```python
from sklearn.metrics import mean_absolute_error

def custom_grid_search(model, hparams, default_kwargs, X_train_cupy, y_train_cupy,
                       folds=3, verbose=0):
    import itertools
    
    # note that the KFold class in SciKit by default does not use shuffling, so we will not implement shuffling here
    X_splits = cp.array_split(X_train_cupy, folds)
    y_splits = cp.array_split(y_train_cupy, folds)
    cv_scores = dict()

    # we use the same model instance to avoid RAPIDS crashing
    cuml_model = model(**default_kwargs)
    for hparam_comb in itertools.product(*hparams.values()):
        kwargs = dict(zip(hparams.keys(), hparam_comb))
        
        holdouts = []
        
        if verbose >= 1:
            print(f'Training CV with {folds} folds on hparams {kwargs}')

        # setting new params seems more stable on CuML
        cuml_model.set_params(**kwargs, **default_kwargs)
        
        for i in range(folds):
            
            # train data is everything except hold-out
            train_sets = [X_splits[j] for j in range(len(X_splits)) if j != i]
            label_sets = [y_splits[j] for j in range(len(y_splits)) if j != i]
                
            train_arr = cp.vstack(train_sets)
            labels_arr = cp.hstack(label_sets)
            cuml_model.fit(train_arr, labels_arr)
            
            # get pred error using hold-out
            preds = cuml_model.predict(X_splits[i])   
            score = mean_absolute_error(cp.asnumpy(preds), cp.asnumpy(y_splits[i]))     
            holdouts.append(score)
            
            if verbose >= 2:
                print(f'\tholdout {i}: {score}')
            
            del train_arr, labels_arr, train_sets, label_sets

        cv_score = np.mean(holdouts)
        cv_scores[str(kwargs)] = cv_score
        
        if verbose >= 1:
            print('CV Score:', cv_score)
            
    del X_splits, y_splits
    
    return cv_scores
```

We will also want to investigate whether our sample size is sufficient for learning, and whether more data would aid learning substantially. Hence, we create a learning curve plotting function. Luckily, Sklearn's `learning_curve` function seems fairly stable with RAPIDS.


```python
def get_learning_curve(estimator, X, y, model_name='Model', train_sizes=[500, 5000, 10000, 20000, 30000, 40000, 50000, 56000]):
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt

    train_sizes, train_scores, test_scores = learning_curve(
        estimator=estimator,
        X=X,
        y=y,
        scoring="neg_mean_absolute_error",
        train_sizes = train_sizes,
        n_jobs=None
    )

    train_mean = -train_scores.mean(axis=1)
    test_mean = -test_scores.mean(axis=1)

    plt.subplots(figsize=(5,4))
    plt.plot(train_sizes, train_mean, label="train")
    plt.plot(train_sizes, test_mean, label="validation")

    plt.title(f'Learning Curve for {model_name}')
    plt.xlabel('Training Set Size')
    plt.ylabel('MAE')
    plt.legend(loc='best')

    plt.show()
```

Finally, we set our device to GPU for accelerated testing.


```python
from cuml.common.device_selection import using_device_type, set_global_device_type
import pickle
set_global_device_type('GPU')
```

### Linear Regression

Each of our ordinal variables seemed to have a (weak) close-to-linear relationship with price. So, using Linear Regression as our first model seems reasonable. We will also not be using Polynomal features for the same reason.

Unfortunately CuML does not natively support MAE Loss for linear regression. However, we can still perfrom mini-batch SGD linear regression and measure MAE loss after-the-fact. We expect this will increase the end MAE loss, and thus negatively affect the model's performance. However, the sklearn implementation's large train time is infeasible given our computational resources.

**NOTE**: Oftentimes many one-hotted features can negatively impact performance for linear regression. To address this, we will train on the ordinal (not one-hotted) data for linear regression. Later, we will train other models which aren't affected by data with many categorical features (e.g. random forests).


```python
from cuml.linear_model import MBSGDRegressor as cumlMBSGDRegressor
```

We will try no regularization, L1 (LASSO) regularization, and L2 (Ridge) regression. We will also try different learning rates and numbers of epochs.


```python
learning_rates = [1e-3, 5e-4, 1e-4]
penatlies = ['none', 'l1', 'l2']
epochs = [5, 10, 20]
variable_lr = ['constant', 'adaptive']

hparam_grid = dict(
    learning_rate=variable_lr,
    eta0=learning_rates,
    penalty=penatlies,
    epochs=epochs,
)

def_kwargs = dict(
    loss='squared_loss',
    tol=0.0,
    fit_intercept=True,
    batch_size=16,
    verbose=False,
)
```


```python
cv_scores_sgd = custom_grid_search(cumlMBSGDRegressor, hparam_grid, def_kwargs, X_train_scaled_cupy[:,:len(ordinal_cols)], y_train_cupy, folds=3, verbose=1)
```

    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.001, 'penalty': 'none', 'epochs': 5}
    CV Score: 14403.76738693239
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.001, 'penalty': 'none', 'epochs': 10}
    CV Score: 13731.922702740245
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.001, 'penalty': 'none', 'epochs': 20}
    CV Score: 13735.842440235096
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.001, 'penalty': 'l1', 'epochs': 5}
    CV Score: 14403.767361263963
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.001, 'penalty': 'l1', 'epochs': 10}
    CV Score: 13731.92264526168
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.001, 'penalty': 'l1', 'epochs': 20}
    CV Score: 13735.84239620721
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.001, 'penalty': 'l2', 'epochs': 5}
    CV Score: 14403.729347750179
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.001, 'penalty': 'l2', 'epochs': 10}
    CV Score: 13731.77704944797
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.001, 'penalty': 'l2', 'epochs': 20}
    CV Score: 13735.774906471992
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'none', 'epochs': 5}
    CV Score: 13999.366175365923
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'none', 'epochs': 10}
    CV Score: 13689.968811665962
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'none', 'epochs': 20}
    CV Score: 13726.747814436421
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'l1', 'epochs': 5}
    CV Score: 13999.366165879655
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'l1', 'epochs': 10}
    CV Score: 13689.968836693151
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'l1', 'epochs': 20}
    CV Score: 13726.7478050614
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'l2', 'epochs': 5}
    CV Score: 13999.454543386712
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'l2', 'epochs': 10}
    CV Score: 13689.977600031285
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'l2', 'epochs': 20}
    CV Score: 13726.727787133786
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0001, 'penalty': 'none', 'epochs': 5}
    CV Score: 13887.24900711396
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0001, 'penalty': 'none', 'epochs': 10}
    CV Score: 13769.577081829084
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0001, 'penalty': 'none', 'epochs': 20}
    CV Score: 13723.805726330704
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0001, 'penalty': 'l1', 'epochs': 5}
    CV Score: 13887.24899779503
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0001, 'penalty': 'l1', 'epochs': 10}
    CV Score: 13769.577080489837
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0001, 'penalty': 'l1', 'epochs': 20}
    CV Score: 13723.805759365072
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0001, 'penalty': 'l2', 'epochs': 5}
    CV Score: 13887.268410754654
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0001, 'penalty': 'l2', 'epochs': 10}
    CV Score: 13769.579302779899
    Training CV with 3 folds on hparams {'learning_rate': 'constant', 'eta0': 0.0001, 'penalty': 'l2', 'epochs': 20}
    CV Score: 13723.80447312882
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.001, 'penalty': 'none', 'epochs': 5}
    CV Score: 14403.76738693239
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.001, 'penalty': 'none', 'epochs': 10}
    CV Score: 13731.922702740245
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.001, 'penalty': 'none', 'epochs': 20}
    CV Score: 13735.842440235096
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.001, 'penalty': 'l1', 'epochs': 5}
    CV Score: 14403.767361263963
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.001, 'penalty': 'l1', 'epochs': 10}
    CV Score: 13731.92264526168
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.001, 'penalty': 'l1', 'epochs': 20}
    CV Score: 13735.84239620721
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.001, 'penalty': 'l2', 'epochs': 5}
    CV Score: 14403.729347750179
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.001, 'penalty': 'l2', 'epochs': 10}
    CV Score: 13731.77704944797
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.001, 'penalty': 'l2', 'epochs': 20}
    CV Score: 13735.774906471992
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0005, 'penalty': 'none', 'epochs': 5}
    CV Score: 13999.366175365923
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0005, 'penalty': 'none', 'epochs': 10}
    CV Score: 13689.968811665962
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0005, 'penalty': 'none', 'epochs': 20}
    CV Score: 13726.747814436421
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0005, 'penalty': 'l1', 'epochs': 5}
    CV Score: 13999.366165879655
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0005, 'penalty': 'l1', 'epochs': 10}
    CV Score: 13689.968836693151
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0005, 'penalty': 'l1', 'epochs': 20}
    CV Score: 13726.7478050614
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0005, 'penalty': 'l2', 'epochs': 5}
    CV Score: 13999.454543386712
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0005, 'penalty': 'l2', 'epochs': 10}
    CV Score: 13689.977600031285
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0005, 'penalty': 'l2', 'epochs': 20}
    CV Score: 13726.727787133786
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0001, 'penalty': 'none', 'epochs': 5}
    CV Score: 13887.24900711396
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0001, 'penalty': 'none', 'epochs': 10}
    CV Score: 13769.577081829084
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0001, 'penalty': 'none', 'epochs': 20}
    CV Score: 13723.805726330704
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0001, 'penalty': 'l1', 'epochs': 5}
    CV Score: 13887.24899779503
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0001, 'penalty': 'l1', 'epochs': 10}
    CV Score: 13769.577080489837
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0001, 'penalty': 'l1', 'epochs': 20}
    CV Score: 13723.805759365072
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0001, 'penalty': 'l2', 'epochs': 5}
    CV Score: 13887.268410754654
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0001, 'penalty': 'l2', 'epochs': 10}
    CV Score: 13769.579302779899
    Training CV with 3 folds on hparams {'learning_rate': 'adaptive', 'eta0': 0.0001, 'penalty': 'l2', 'epochs': 20}
    CV Score: 13723.80447312882



```python
opt_sgd_params = min(cv_scores_sgd, key=cv_scores_sgd.get)

print(opt_sgd_params)
```

    {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'none', 'epochs': 10}



```python
opt_sgd_params = {'learning_rate': 'constant', 'eta0': 0.0005, 'penalty': 'l2', 'epochs': 10}
def_kwargs['verbose'] = 2
opt_sgd_model = cumlMBSGDRegressor(**opt_sgd_params, **def_kwargs)

opt_sgd_model.fit(X_train_scaled_cupy[:,:len(ordinal_cols)], y_train_cupy)
```


    MBSGDRegressor()



```python
opt_sgd_preds = opt_sgd_model.predict(X_test_scaled_cupy[:,:len(ordinal_cols)])   
opt_sgd_score = mean_absolute_error(cp.asnumpy(opt_sgd_preds), cp.asnumpy(y_test_cupy))     
print('MBSGDRegressor MAE:', opt_sgd_score)
```

    MBSGDRegressor MAE: 13706.061317947917


When examining our learning curve, it seems that, although MAE overall decreases somewhat with more data, this is not a strict decline; while new data might help make better predictions, it can also inject more noise. Perhaps, for our Linear Regressor in particular, using the full 500k samples could allow the model to decrease MAE marginally, but it seems more data is not necessarily the solution for better performance.

Therefore, we will proceed to running more powerful estimators to see if we can improve performance.


```python
get_learning_curve(opt_sgd_model, X=cp.asnumpy(X_train_scaled_cupy[:,:len(ordinal_cols)]), y=cp.asnumpy(y_train_cupy), model_name='MBSGDRegressor')
```


    
![png](./img/output_60_0.png)
    


### KNN Regression

KNNs can also be used for regression and don't require training a model (since we only use data). Because we have a very large dataset, it KNN regression might perform well.

That being said, KNNs are also sensitive to unscaled data, as well as data with many features (due to the curse of dimensionality). Therefore, we will used only the scaled, ordinal data here as well.


```python
from cuml.neighbors import KNeighborsRegressor
```

Due to the simplicity of KNN, we only need to test the k-values in our hyperparam search. Since the KNN regressor requires no real train time, we can test many k-values in our grid search.


```python
n_neighbors = list(np.arange(1, 100))

hparam_grid = dict(
    n_neighbors=n_neighbors,
)

def_kwargs = dict(
    verbose=False,
)
```


```python
cv_scores_knn = custom_grid_search(KNeighborsRegressor, hparam_grid, def_kwargs, X_train_scaled_cupy[:,:len(ordinal_cols)], y_train_cupy, folds=3, verbose=1)
```

    Training CV with 3 folds on hparams {'n_neighbors': 1}
    CV Score: 15975.834725993043
    Training CV with 3 folds on hparams {'n_neighbors': 2}
    CV Score: 14492.496787926195
    Training CV with 3 folds on hparams {'n_neighbors': 3}
    CV Score: 13854.731727652039
    Training CV with 3 folds on hparams {'n_neighbors': 4}
    CV Score: 13558.999188825675
    Training CV with 3 folds on hparams {'n_neighbors': 5}
    CV Score: 13363.38026333578
    Training CV with 3 folds on hparams {'n_neighbors': 6}
    CV Score: 13218.484661646899
    Training CV with 3 folds on hparams {'n_neighbors': 7}
    CV Score: 13122.148907950303
    Training CV with 3 folds on hparams {'n_neighbors': 8}
    CV Score: 13044.330554730457
    Training CV with 3 folds on hparams {'n_neighbors': 9}
    CV Score: 12994.160171705907
    Training CV with 3 folds on hparams {'n_neighbors': 10}
    CV Score: 12954.936878527145
    Training CV with 3 folds on hparams {'n_neighbors': 11}
    CV Score: 12918.059344215197
    Training CV with 3 folds on hparams {'n_neighbors': 12}
    CV Score: 12893.230926293496
    Training CV with 3 folds on hparams {'n_neighbors': 13}
    CV Score: 12871.036419000375
    Training CV with 3 folds on hparams {'n_neighbors': 14}
    CV Score: 12847.819706001814
    Training CV with 3 folds on hparams {'n_neighbors': 15}
    CV Score: 12836.641517834832
    Training CV with 3 folds on hparams {'n_neighbors': 16}
    CV Score: 12831.496256877059
    Training CV with 3 folds on hparams {'n_neighbors': 17}
    CV Score: 12820.931408824805
    Training CV with 3 folds on hparams {'n_neighbors': 18}
    CV Score: 12809.298053431608
    Training CV with 3 folds on hparams {'n_neighbors': 19}
    CV Score: 12803.781044895608
    Training CV with 3 folds on hparams {'n_neighbors': 20}
    CV Score: 12801.882016682706
    Training CV with 3 folds on hparams {'n_neighbors': 21}
    CV Score: 12795.48541643584
    Training CV with 3 folds on hparams {'n_neighbors': 22}
    CV Score: 12797.623877092597
    Training CV with 3 folds on hparams {'n_neighbors': 23}
    CV Score: 12793.723181260111
    Training CV with 3 folds on hparams {'n_neighbors': 24}
    CV Score: 12788.474411899486
    Training CV with 3 folds on hparams {'n_neighbors': 25}
    CV Score: 12788.14960382096
    Training CV with 3 folds on hparams {'n_neighbors': 26}
    CV Score: 12790.386931446323
    Training CV with 3 folds on hparams {'n_neighbors': 27}
    CV Score: 12789.337931022099
    Training CV with 3 folds on hparams {'n_neighbors': 28}
    CV Score: 12794.82659252065
    Training CV with 3 folds on hparams {'n_neighbors': 29}
    CV Score: 12789.80166992514
    Training CV with 3 folds on hparams {'n_neighbors': 30}
    CV Score: 12787.542603144742
    Training CV with 3 folds on hparams {'n_neighbors': 31}
    CV Score: 12792.504355774523
    Training CV with 3 folds on hparams {'n_neighbors': 32}
    CV Score: 12793.113535565377
    Training CV with 3 folds on hparams {'n_neighbors': 33}
    CV Score: 12795.498605527106
    Training CV with 3 folds on hparams {'n_neighbors': 34}
    CV Score: 12794.273839276875
    Training CV with 3 folds on hparams {'n_neighbors': 35}
    CV Score: 12798.359146351197
    Training CV with 3 folds on hparams {'n_neighbors': 36}
    CV Score: 12803.41079282206
    Training CV with 3 folds on hparams {'n_neighbors': 37}
    CV Score: 12806.414753237523
    Training CV with 3 folds on hparams {'n_neighbors': 38}
    CV Score: 12807.1858425581
    Training CV with 3 folds on hparams {'n_neighbors': 39}
    CV Score: 12803.362864560891
    Training CV with 3 folds on hparams {'n_neighbors': 40}
    CV Score: 12805.262657515292
    Training CV with 3 folds on hparams {'n_neighbors': 41}
    CV Score: 12807.428284061389
    Training CV with 3 folds on hparams {'n_neighbors': 42}
    CV Score: 12806.243849465376
    Training CV with 3 folds on hparams {'n_neighbors': 43}
    CV Score: 12803.454128659834
    Training CV with 3 folds on hparams {'n_neighbors': 44}
    CV Score: 12809.268323666658
    Training CV with 3 folds on hparams {'n_neighbors': 45}
    CV Score: 12810.016002848008
    Training CV with 3 folds on hparams {'n_neighbors': 46}
    CV Score: 12807.448404760216
    Training CV with 3 folds on hparams {'n_neighbors': 47}
    CV Score: 12810.157341759432
    Training CV with 3 folds on hparams {'n_neighbors': 48}
    CV Score: 12813.986476399048
    Training CV with 3 folds on hparams {'n_neighbors': 49}
    CV Score: 12816.193100192235
    Training CV with 3 folds on hparams {'n_neighbors': 50}
    CV Score: 12813.895172935207
    Training CV with 3 folds on hparams {'n_neighbors': 51}
    CV Score: 12815.110918959728
    Training CV with 3 folds on hparams {'n_neighbors': 52}
    CV Score: 12817.513532741941
    Training CV with 3 folds on hparams {'n_neighbors': 53}
    CV Score: 12816.510682602368
    Training CV with 3 folds on hparams {'n_neighbors': 54}
    CV Score: 12816.407319740174
    Training CV with 3 folds on hparams {'n_neighbors': 55}
    CV Score: 12816.895920655172
    Training CV with 3 folds on hparams {'n_neighbors': 56}
    CV Score: 12819.15144589749
    Training CV with 3 folds on hparams {'n_neighbors': 57}
    CV Score: 12821.696325562012
    Training CV with 3 folds on hparams {'n_neighbors': 58}
    CV Score: 12823.685939718567
    Training CV with 3 folds on hparams {'n_neighbors': 59}
    CV Score: 12824.656584170456
    Training CV with 3 folds on hparams {'n_neighbors': 60}
    CV Score: 12825.021699635088
    Training CV with 3 folds on hparams {'n_neighbors': 61}
    CV Score: 12825.941413995723
    Training CV with 3 folds on hparams {'n_neighbors': 62}
    CV Score: 12828.438727739058
    Training CV with 3 folds on hparams {'n_neighbors': 63}
    CV Score: 12829.218575183882
    Training CV with 3 folds on hparams {'n_neighbors': 64}
    CV Score: 12830.323126373922
    Training CV with 3 folds on hparams {'n_neighbors': 65}
    CV Score: 12833.769018994595
    Training CV with 3 folds on hparams {'n_neighbors': 66}
    CV Score: 12835.855644872201
    Training CV with 3 folds on hparams {'n_neighbors': 67}
    CV Score: 12837.57230168015
    Training CV with 3 folds on hparams {'n_neighbors': 68}
    CV Score: 12837.100842234067
    Training CV with 3 folds on hparams {'n_neighbors': 69}
    CV Score: 12837.211644849094
    Training CV with 3 folds on hparams {'n_neighbors': 70}
    CV Score: 12838.00709537686
    Training CV with 3 folds on hparams {'n_neighbors': 71}
    CV Score: 12841.385466373787
    Training CV with 3 folds on hparams {'n_neighbors': 72}
    CV Score: 12842.675566863874
    Training CV with 3 folds on hparams {'n_neighbors': 73}
    CV Score: 12844.31466981169
    Training CV with 3 folds on hparams {'n_neighbors': 74}
    CV Score: 12845.967049130122
    Training CV with 3 folds on hparams {'n_neighbors': 75}
    CV Score: 12845.287119465143
    Training CV with 3 folds on hparams {'n_neighbors': 76}
    CV Score: 12846.574629043627
    Training CV with 3 folds on hparams {'n_neighbors': 77}
    CV Score: 12847.241874916494
    Training CV with 3 folds on hparams {'n_neighbors': 78}
    CV Score: 12849.2924989448
    Training CV with 3 folds on hparams {'n_neighbors': 79}
    CV Score: 12850.657142222219
    Training CV with 3 folds on hparams {'n_neighbors': 80}
    CV Score: 12851.376037602733
    Training CV with 3 folds on hparams {'n_neighbors': 81}
    CV Score: 12851.919400113251
    Training CV with 3 folds on hparams {'n_neighbors': 82}
    CV Score: 12851.373827063968
    Training CV with 3 folds on hparams {'n_neighbors': 83}
    CV Score: 12852.706441280514
    Training CV with 3 folds on hparams {'n_neighbors': 84}
    CV Score: 12853.03610690362
    Training CV with 3 folds on hparams {'n_neighbors': 85}
    CV Score: 12853.61913676728
    Training CV with 3 folds on hparams {'n_neighbors': 86}
    CV Score: 12854.79721053105
    Training CV with 3 folds on hparams {'n_neighbors': 87}
    CV Score: 12857.08774461218
    Training CV with 3 folds on hparams {'n_neighbors': 88}
    CV Score: 12857.441761983491
    Training CV with 3 folds on hparams {'n_neighbors': 89}
    CV Score: 12858.7084177897
    Training CV with 3 folds on hparams {'n_neighbors': 90}
    CV Score: 12859.859954017562
    Training CV with 3 folds on hparams {'n_neighbors': 91}
    CV Score: 12860.84864928652
    Training CV with 3 folds on hparams {'n_neighbors': 92}
    CV Score: 12862.661032855214
    Training CV with 3 folds on hparams {'n_neighbors': 93}
    CV Score: 12864.114873847488
    Training CV with 3 folds on hparams {'n_neighbors': 94}
    CV Score: 12865.449639067489
    Training CV with 3 folds on hparams {'n_neighbors': 95}
    CV Score: 12869.031862011749
    Training CV with 3 folds on hparams {'n_neighbors': 96}
    CV Score: 12871.530233257998
    Training CV with 3 folds on hparams {'n_neighbors': 97}
    CV Score: 12871.793247876923
    Training CV with 3 folds on hparams {'n_neighbors': 98}
    CV Score: 12872.546701344494
    Training CV with 3 folds on hparams {'n_neighbors': 99}
    CV Score: 12873.230274156484



```python
opt_knn_params = min(cv_scores_knn, key=cv_scores_knn.get)

print(opt_knn_params)
```

    {'n_neighbors': 30}



```python
opt_knn_params = {'n_neighbors': 30}
def_kwargs['verbose'] = 2
opt_knn_model = KNeighborsRegressor(**opt_knn_params, **def_kwargs)

opt_knn_model.fit(X_train_scaled_cupy[:,:len(ordinal_cols)], y_train_cupy)
```


    KNeighborsRegressor()


We are able to acheive lower MAE loss than our Linear Regression!


```python
opt_knn_preds = opt_knn_model.predict(X_test_cupy[:,:len(ordinal_cols)])   
opt_knn_score = mean_absolute_error(cp.asnumpy(opt_knn_preds), cp.asnumpy(y_test_cupy))     
print('KNeighborsRegressor MAE:', opt_knn_score)
```

    KNeighborsRegressor MAE: 12684.34772671146


While the learning curve sees a more stable decrease in MAE with more data, the end result is similar as before: more data could allow the model to decrease MAE somewhat, but not by a significant amount.


```python
get_learning_curve(opt_knn_model, X=cp.asnumpy(X_train_scaled_cupy[:,:len(ordinal_cols)]), y=cp.asnumpy(y_train_cupy), model_name='KNeighborsRegressor')
```

    [I] [12:51:10.570436] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:10.602495] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:10.631667] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:10.696755] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:10.789240] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:10.859824] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:10.927509] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.003513] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.093514] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.100216] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.109746] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.124332] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.149680] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.183933] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.231521] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.299927] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.381469] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.388648] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.398499] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.411411] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.434696] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.468809] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.522114] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.591567] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.671812] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.680542] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.691674] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.706529] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.730910] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.764291] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.813283] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.880655] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.964183] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.971687] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.983541] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:11.998982] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:12.021375] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:12.056714] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:12.111369] Unused keyword parameter: n_jobs during cuML estimator initialization
    [I] [12:51:12.183556] Unused keyword parameter: n_jobs during cuML estimator initialization



    
![png](./img/output_72_1.png)
    


### Random Forests for Regression

Next, we implement Random Forests for regression. Random Forests are an ensemble method which perform well on a host of different problems. Additionally, unlike models like linear regression or KNN regression, random forests are able to handle unscaled and categorical data well. So, we use the whole dataset here.


```python
from cuml.ensemble import RandomForestRegressor as cuRF
```


```python
n_estimators = [100, 300, 500, 600]
max_depth = [10, 40, 100]
min_samples_split = [2, 5, 10]

hparam_grid = dict(
    n_estimators=n_estimators,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
)

def_kwargs = dict(
    accuracy_metric='mean_ae',
    verbose=False,
    random_state=SEED,
    n_streams=1,
)
```


```python
cv_scores_RF = custom_grid_search(cuRF, hparam_grid, def_kwargs, X_train_cupy, y_train_cupy, folds=3, verbose=1)
```

    Training CV with 3 folds on hparams {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 2}
    CV Score: 12262.603246456178
    Training CV with 3 folds on hparams {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5}
    CV Score: 12258.12945560379
    Training CV with 3 folds on hparams {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 10}
    CV Score: 12250.63613764298
    Training CV with 3 folds on hparams {'n_estimators': 100, 'max_depth': 40, 'min_samples_split': 2}
    CV Score: 12050.330507251123
    Training CV with 3 folds on hparams {'n_estimators': 100, 'max_depth': 40, 'min_samples_split': 5}
    CV Score: 11982.79636236684
    Training CV with 3 folds on hparams {'n_estimators': 100, 'max_depth': 40, 'min_samples_split': 10}
    CV Score: 11896.202565839785
    Training CV with 3 folds on hparams {'n_estimators': 100, 'max_depth': 100, 'min_samples_split': 2}
    CV Score: 12050.330782143443
    Training CV with 3 folds on hparams {'n_estimators': 100, 'max_depth': 100, 'min_samples_split': 5}
    CV Score: 11982.796468562554
    Training CV with 3 folds on hparams {'n_estimators': 100, 'max_depth': 100, 'min_samples_split': 10}
    CV Score: 11896.202565839785
    Training CV with 3 folds on hparams {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 2}
    CV Score: 12267.933520535982
    Training CV with 3 folds on hparams {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 5}
    CV Score: 12262.761485227695
    Training CV with 3 folds on hparams {'n_estimators': 300, 'max_depth': 10, 'min_samples_split': 10}
    CV Score: 12253.534869872516
    Training CV with 3 folds on hparams {'n_estimators': 300, 'max_depth': 40, 'min_samples_split': 2}
    CV Score: 12027.358765933304
    Training CV with 3 folds on hparams {'n_estimators': 300, 'max_depth': 40, 'min_samples_split': 5}
    CV Score: 11960.204914837144
    Training CV with 3 folds on hparams {'n_estimators': 300, 'max_depth': 40, 'min_samples_split': 10}
    CV Score: 11878.274885952933
    Training CV with 3 folds on hparams {'n_estimators': 300, 'max_depth': 100, 'min_samples_split': 2}
    CV Score: 12027.358713812026
    Training CV with 3 folds on hparams {'n_estimators': 300, 'max_depth': 100, 'min_samples_split': 5}
    CV Score: 11960.205226113912
    Training CV with 3 folds on hparams {'n_estimators': 300, 'max_depth': 100, 'min_samples_split': 10}
    CV Score: 11878.275374799201
    Training CV with 3 folds on hparams {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2}
    CV Score: 12265.04699352866
    Training CV with 3 folds on hparams {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 5}
    CV Score: 12260.445383052807
    Training CV with 3 folds on hparams {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 10}
    CV Score: 12250.445806831092
    Training CV with 3 folds on hparams {'n_estimators': 500, 'max_depth': 40, 'min_samples_split': 2}
    CV Score: 12017.915814656832
    Training CV with 3 folds on hparams {'n_estimators': 500, 'max_depth': 40, 'min_samples_split': 5}
    CV Score: 11952.663534637753
    Training CV with 3 folds on hparams {'n_estimators': 500, 'max_depth': 40, 'min_samples_split': 10}
    CV Score: 11871.07265968489
    Training CV with 3 folds on hparams {'n_estimators': 500, 'max_depth': 100, 'min_samples_split': 2}
    CV Score: 12017.915574251829
    Training CV with 3 folds on hparams {'n_estimators': 500, 'max_depth': 100, 'min_samples_split': 5}
    CV Score: 11952.66371304432
    Training CV with 3 folds on hparams {'n_estimators': 500, 'max_depth': 100, 'min_samples_split': 10}
    CV Score: 11871.072952992652
    Training CV with 3 folds on hparams {'n_estimators': 600, 'max_depth': 10, 'min_samples_split': 2}
    CV Score: 12264.34822568825
    Training CV with 3 folds on hparams {'n_estimators': 600, 'max_depth': 10, 'min_samples_split': 5}
    CV Score: 12260.052136386306
    Training CV with 3 folds on hparams {'n_estimators': 600, 'max_depth': 10, 'min_samples_split': 10}
    CV Score: 12249.919887799508
    Training CV with 3 folds on hparams {'n_estimators': 600, 'max_depth': 40, 'min_samples_split': 2}
    CV Score: 12015.887775787274
    Training CV with 3 folds on hparams {'n_estimators': 600, 'max_depth': 40, 'min_samples_split': 5}
    CV Score: 11951.606848569836
    Training CV with 3 folds on hparams {'n_estimators': 600, 'max_depth': 40, 'min_samples_split': 10}
    CV Score: 11870.515225107534
    Training CV with 3 folds on hparams {'n_estimators': 600, 'max_depth': 100, 'min_samples_split': 2}
    CV Score: 12015.886890712776
    Training CV with 3 folds on hparams {'n_estimators': 600, 'max_depth': 100, 'min_samples_split': 5}
    CV Score: 11951.60714634382
    Training CV with 3 folds on hparams {'n_estimators': 600, 'max_depth': 100, 'min_samples_split': 10}
    CV Score: 11870.515664901764



```python
opt_rf_params = min(cv_scores_RF, key=cv_scores_RF.get)

print(opt_rf_params)
```

    {'n_estimators': 600, 'max_depth': 40, 'min_samples_split': 10}



```python
opt_rf_params = {'n_estimators': 600, 'max_depth': 40, 'min_samples_split': 10}
def_kwargs['verbose'] = 2
opt_rf_model = cuRF(**opt_rf_params, **def_kwargs)

opt_rf_model.fit(X_train_cupy, y_train_cupy)
```


    RandomForestRegressor()


As seen below, we achieve lower MAE compared to Linear Regression or KNN regression!


```python
opt_rf_preds = opt_rf_model.predict(X_test_cupy)   
opt_rf_score = mean_absolute_error(cp.asnumpy(opt_rf_preds), cp.asnumpy(y_test_cupy))     
print('RandomForest MAE:', opt_rf_score)
```

    RandomForest MAE: 11737.414786409116



```python
opt_rf_params = {'n_estimators': 100, 'max_depth': 40, 'min_samples_split': 10}
def_kwargs['verbose'] = 2
opt_rf_model = cuRF(**opt_rf_params, **def_kwargs)
```

Our Random Forest does overall perform better, but our loss decreases seems like it is leveling off as previous models.


```python
get_learning_curve(opt_rf_model, X=cp.asnumpy(X_train_cupy), y=cp.asnumpy(y_train_cupy), model_name='RandomForest', train_sizes=[500, 5000, 15000, 35000, 50000, 56000])
```


    
![png](./img/output_84_0.png)
    


### XGBoost Regression

XGBoost is one of the best-performing classical ML algorithms. Here, we will *not* be using RAPIDS; instead be will be using the [xgboost](https://xgboost.readthedocs.io/en/stable/) library.

This library seems much more stable on our machines, and works well with Sklearn's hyperparam search classes. However, there are many hyperparams to test, and some combinations can take a while to train (both because XGBoost is somewhat more complicated than KNN, for example, as well as because CuML seems faster than using Sklearn grid search with `xgboost`). So, we used `RandomizedSearchCV` instead of grid search.


```python
from xgboost import XGBRegressor

xg_boost_model = XGBRegressor(
    objective= 'reg:absoluteerror',
    tree_method='gpu_hist',
    nthread=4,
    seed=SEED,
)

params = {
    'learning_rate': [0.1, 0.01, 0.05],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'max_depth': range(2, 10, 1),
    'n_estimators': [100, 500, 1000, 2000],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'subsample': [0.6, 0.8, 1.0],
}
```


```python
from sklearn.model_selection import RandomizedSearchCV

xg_rand_search = RandomizedSearchCV(xg_boost_model, params, n_iter=100,
                                    scoring='neg_mean_absolute_error', cv=3,
                                    random_state=SEED, verbose=1, refit=True,)
```


```python
xg_rand_search.fit(X_train, y_train)
```

    Fitting 3 folds for each of 100 candidates, totalling 300 fits




```python
xg_rand_search.best_params_
```


    {'subsample': 0.8,
     'n_estimators': 2000,
     'max_depth': 2,
     'learning_rate': 0.1,
     'gamma': 2,
     'colsample_bytree': 0.6}



```python
opt_xg_boost_model = XGBRegressor(
    objective= 'reg:absoluteerror',
    tree_method='gpu_hist',
    nthread=4,
    seed=SEED,
    # opt parameters
    subsample=0.8,
    n_estimators=2000,
    max_depth=2,
    learning_rate=0.1,
    gamma=2,
    colsample_bytree=0.6
)

opt_xg_boost_model.fit(X_train, y_train)
```


```python
opt_xg_boost_preds = opt_xg_boost_model.predict(X_test)
opt_xg_boost_score = mean_absolute_error(opt_xg_boost_preds, y_test)
print('XGBoostRegressor MAE:', opt_xg_boost_score)
```

    XGBoostRegressor MAE: 11922.813671071091


XGBoost has a similar situation with its learning curve: more data is unlikely to significantly decrease MAE.


```python
get_learning_curve(opt_xg_boost_model, X=X_train, y=y_train, model_name='RandomForest')
```


    
![png](./img/output_94_0.png)
    


### DNN For Regression

The above methods all used classical ML methods. Now, we examine whether DL can help better tackle our regression problem.

Below, we train a DNN for regression using PyTorch.


```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
```


```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

device
```


    device(type='mps')


Below we configure a PyTorch Dataset. Later, we will use Dataloaders for efficient batch loading for our data.


```python
class RealEstateDataset(Dataset):
    def __init__(self, X_train, y_train, X_test, y_test, train=True, max_cache_size=800000):
        self.df = df
        self.train = train

        self.X = X_train if self.train else X_test
        self.y = y_train if self.train else y_test

        self.cache = dict()
        self.max_cache_size = max_cache_size

    def __getitem__(self, index):

        if index in self.cache.keys():
            return self.cache[index]

        X_sample = torch.tensor(self.X[index]).to(torch.float32).squeeze()
        y_sample = torch.tensor(self.y[index]).to(torch.float32).squeeze()

        if len(self.cache) >= self.max_cache_size:
            self.cache.popitem()

        self.cache[index] = (X_sample, y_sample)

        return self.cache[index]

    def __len__(self):
        return len(self.X)
```


```python
train_dataset = RealEstateDataset(X_train_scaled.to_numpy(), y_train.to_numpy(), X_test_scaled.to_numpy(), y_test.to_numpy(), train=True)
test_dataset = RealEstateDataset(X_train_scaled.to_numpy(), y_train.to_numpy(), X_test_scaled.to_numpy(), y_test.to_numpy(), train=False)
```

Now we buid our DNN. We use the following considerations:

1. We try using both regular and leaky ReLU. While tanh and sigmoid we used for a while in DL, the modern consensus is that [they do not perform as well as simple ReLUs (and their variants)](https://www.aitude.com/comparison-of-sigmoid-tanh-and-relu-activation-functions/).
2. We offer an optional implementation for dropout (since they can help improve generalization), but in our testing it simply increases the number of epochs needed to acheive optimality.
3. We use He initialization for the weights, the most common initialization for ReLU activatons. Brief testing inicated that fan-in performed better, meaning our forward-pass variances were likely more "chaotic" than for our backward passes (most likely because the initial forward pass is performed without any fitting to the data).

NOTE: Normally, we'd use a separate validation set for hyperparam tuning, then retrain on train and validation, then test the model on the test set. However, for DNNs, this is not feasible since they take much longer to train. So, instead, we simply use a train and test set.


```python
class RegressionDNN(nn.Module):
    def __init__(self, input_size, fcs=[24, 12, 6, 1], dropout=None, relu='leaky'):
        super(RegressionDNN, self).__init__()

        self.input_size = input_size
        self.fcs = fcs
        self.dropout = dropout
        self.relu = relu

        self.model = nn.Sequential(
            *self._make_layers()
        )

        self.model.apply(self.init_weights)

    def _make_layers(self):
        layers = [nn.Linear(self.input_size, self.fcs[0])]

        for i in range(len(self.fcs) - 1):
            if self.relu == 'normal':
                layers.append(nn.ReLU())
            elif self.relu == 'leaky':
                layers.append(nn.LeakyReLU())
            if self.dropout is not None:
                layers.append(nn.Dropout(self.dropout))
            layers.append(nn.Linear(self.fcs[i], self.fcs[i+1]))

        return layers
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            m.bias.data.fill_(0.01)

    def forward(self, x):
        return self.model(x).squeeze()
```


```python
BATCH_SIZE = 64
EPOCHS = 500
DATA_DIMS = train_dataset[0][0].size(0)
```


```python
train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE)
```

Our train pass includes both a train and validation iteration so we can evaluate generalization as training proceeds (so we can determine when our DNN starts to overfit).

We also save our model every epoch so we can load the optimal weights after training concliudes (unlike the other models, training the DNN is quite slow).


```python
from tqdm import tqdm
import sys
from pathlib import Path
import os
import wandb

def save(model, optimizer, save_path='model_checkpoint.pt'):

    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(save_dict, save_path)

def load(model, optimizer, load_path='model_checkpoint.pt'):

    checkpoint = torch.load(load_path, map_location=device)

    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    return model, optimizer

def train(train_dl, test_dl, epochs=EPOCHS, batch_size=BATCH_SIZE, input_size=DATA_DIMS, print_batch_every=None,
          lr=1e-4, dropout=None, opt='adam', relu='normal',
          checkpoint_dir='./checkpoints', pretrained_path=None,
          logging = False, log_init = False, project_name='Russian-Real-Estate-Regression', group_name='DNN', run_name=None,
          model_fcs=None):
    
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if logging and log_init:
        wandb.init(project=project_name, group=group_name, name=run_name, config=dict(
            batch_size = batch_size,
            epochs = epochs,
            fcs = str(model_fcs),
            lr = lr,
            dropout=dropout,
            ReLU=relu,
            opt=opt,
        ))

    model = RegressionDNN(input_size, fcs=model_fcs, dropout=dropout, relu=relu).to(device)
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if pretrained_path is not None:
        model, optimizer = load(model, optimizer, load_path=pretrained_path)

    train_losses = []
    test_losses = []

    for epoch in tqdm(range(epochs)):

        model.train()

        batch = 0
        train_loss = 0
        for data in iter(train_dl):
            batch += 1

            X, y = data

            X = X.to(device)
            y = y.to(device)
            
            optimizer.zero_grad()

            pred = model(X)

            loss = F.l1_loss(pred, y * 1e-3, reduction='sum')
            train_loss += loss

            loss.backward()
            optimizer.step()

            if print_batch_every is not None and ((batch-1) % print_batch_every == 0):
                print(f'epoch: {epoch}\tbatch: {batch}/{len(train_dl)}\ttrain_loss: {loss.item()}', file=sys.stderr)

        train_loss /= len(train_dl.dataset)

        model.eval()

        with torch.no_grad():

            batch = 0
            test_loss = 0
            for data in iter(test_dl):
                batch += 1

                X, y = data

                X = X.to(device)
                y = y.to(device)

                pred = model(X)

                loss = F.l1_loss(pred, y * 1e-3, reduction='sum')
                test_loss += loss

                if print_batch_every is not None and ((batch-1) % print_batch_every == 0):
                    print(f'epoch: {epoch}\tbatch: {batch}/{len(test_dl)}\ttest_loss: {loss.item()}', file=sys.stderr)

            test_loss /= len(test_dl.dataset)

        save_path = Path(checkpoint_dir) / Path(f'reg_model_{epoch}.pt')
        save(model, optimizer, save_path=str(save_path))

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if logging:
            wandb.log({ 'train/loss': train_loss * 1e3, 'test/loss': test_loss * 1e3 })

    if logging:
        wandb.finish()

    return model, train_losses, test_losses
```

For our DNN, we can't run a formal hyperparam search due to the time it takes these models to trian. Instead, we did the following:

When selecting hyperparams, we ran 100 epochs using
1. different learning rates
2. AdamW vs Adam optimizer
3. regular ReLU vs leaky ReLU

Then, we selected the best-converging model (`1e-4` learning rate, Adam optimizer, with leaky ReLU) for proper training with 500 epochs.

<table>
    <tr><td colspan='2'>Hparam Testing (100 Epochs)</td></tr>
    <tr>
        <td><a href='https://wandb.ai/arth-shukla/COGS-118A-Russian-Real-Estate-Regression'><img src='./assets/hparam_train_loss.png' /></a></td>
        <td><a href='https://wandb.ai/arth-shukla/COGS-118A-Russian-Real-Estate-Regression'><img src='./assets/hparam_test_loss.png' /></a></td>
    </tr>
    <tr><td colspan='2'>Training (500 Epochs)</td></tr>
    <tr>
        <td><a href='https://wandb.ai/arth-shukla/COGS-118A-Russian-Real-Estate-Regression'><img src='./assets/train_train_loss.png' /></a></td>
        <td><a href='https://wandb.ai/arth-shukla/COGS-118A-Russian-Real-Estate-Regression'><img src='./assets/train_test_loss.png' /></a></td>
    </tr>
</table>

It seems our model quickly converges to an optimal solution in about 100 epochs, after which it begins to overfit. So, we choose the version of our model with lowest generalization error (around epoch 100) for our final DNN model.


```python
for lr in [1e-4, 1e-5, 5e-5]:
    trained_model, train_losses, test_losses = train(train_dl, test_dl, logging=False, log_init=False,
                                project_name='Russian-Real-Estate-Regression', group_name='DNN-Training', run_name=f'adam_leakyrelu_lr={lr}',
                                print_batch_every=None, checkpoint_dir=f'./selected-checkpoints/optimal-dnn.pt',
                                opt='adam', relu='leaky',
                                model_fcs=[128, 256, 256, 256, 1], lr=lr, dropout=None, epochs=500)
```

Finally, we run our model over our test set so we can compare our DNN with our other models.


```python
def eval(test_dl, model_fcs=[128, 256, 256, 256, 1], input_size=DATA_DIMS, dropout=None, pretrained_path=None):
    
    model = RegressionDNN(input_size, fcs=model_fcs, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    model, optimizer = load(model, optimizer, load_path=pretrained_path)
    
    with torch.no_grad():

        all_preds = []
        batch = 0
        test_loss = 0
        for data in iter(test_dl):
            batch += 1

            X, y = data

            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            all_preds.append(pred)

            loss = F.l1_loss(pred, y * 1e-3, reduction='sum')
            test_loss += loss

        test_loss /= len(test_dl.dataset)
        
    return test_loss.item() * 1e3, torch.hstack(all_preds).cpu().numpy() * 1e3
```

Our model seems to perform about in-line Random Forest and XGBoost.


```python
opt_dnn_loss, opt_dnn_preds = eval(test_dl, pretrained_path='./selected_checkpoints/optimal-dnn.pt')

print('Regression DNN MAE:', opt_dnn_loss)
```

    Regression DNN MAE: 11860.313415527344


Note that we will not be plotting a learning curve for the DNN since training and retraining would take too long.

## Performance Analysis of Different Models

Recall the MAE test performance for the optimal hyperparams for each regression algorithm:
- MB SGB Linear Reg: `13706.061317947917`
- Random Forest: `11737.414786409116`
- KNN: `12684.34772671146`
- XGBoost: `11922.813671071091`
- DNN: `11860.313415527344`

Recall that the price data is scaled such that MAE has units of US Dollars.

We now analyze the peformance of each trained model across the price distribution. The below function create a histogram of the average error (not absolute) for intervals across the price distribution. This way, we can evaluate on what types of real estate each model peforms best.


```python
pred_df = pd.DataFrame()
pred_df['Price'] = y_test
pred_df['LinReg'] = opt_sgd_preds
pred_df['RF'] = opt_rf_preds
pred_df['KNN'] = opt_knn_preds
pred_df['XGBoost'] = opt_xg_boost_preds
pred_df['DNN'] = opt_dnn_preds
```


```python
def generate_plots(pred_df,
                models=['LinReg', 'KNN', 'RF', 'XGBoost', 'DNN'],
                rows=2, figsize=(20,10),
                price_bounds=(0, 200000), step=10000):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import math
    from sklearn.metrics import mean_absolute_error

    def mae_by_price(price_df, model='RF', price_bounds=price_bounds, step=step):
        lb, ub = price_bounds

        price_df = price_df[price_df['Price'] >= lb]
        price_df = price_df[price_df['Price'] <= ub]

        price_bounds = range(lb, ub+step, step)
        price_by_range = dict()

        for i in range(len(price_bounds) - 1):
            low = price_bounds[i]
            high = price_bounds[i+1]

            curr_df = price_df.copy()
            curr_df = curr_df[curr_df['Price'] >= low]
            curr_df = curr_df[curr_df['Price'] <= high]

            price_by_range[f'{low}-{high}'] = mean_absolute_error(curr_df[model], curr_df['Price'])
            # price_by_range[f'{low}-{high}'] = np.mean(curr_df[model] - curr_df['Price'])

        price_by_range_df = pd.DataFrame()
        price_by_range_df['Ground Truth Price Range'] = price_by_range.keys()
        price_by_range_df[f'{model} MAE'] = price_by_range.values()

        return price_by_range_df

    cols = math.ceil(len(models) / rows)

    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, sharex=False, sharey=True, figsize=figsize
    )

    ii, jj = -1, 0
    for i, model_name in enumerate(models):
        if i % cols == 0:
            ii += 1
            jj = 0
        else:
            jj += 1
        
        plot_df = mae_by_price(pred_df.copy(), model=model_name)

        sns.barplot(data=plot_df, x=f'{model_name} MAE', y='Ground Truth Price Range', orient='h', ax=axes[ii, jj])
        axes[ii, jj].title.set_text(f'{model_name} MAE by Price')
```

We will analyze the data in two chunks:

1. Housing labeled as less than 120,000 USD, since a majority of our data seems to be on the cheaper end of our distribution.
2. Housing labeled as more than 120,000 USD. 


```python
generate_plots(pred_df.copy(), models=['LinReg', 'KNN', 'RF', 'XGBoost', 'DNN'], price_bounds=(0, 120000), step=5000, rows=2, figsize=(20,10),)
plt.show()
```


    
![png](./img/output_120_0.png)
    


The above histograms indicate that our models all perform best on housing in the $15,000-70,000 range. However, for extremely cheap housing, or for more expensive housing, our models do not perform as well.


```python
generate_plots(pred_df, models=['LinReg', 'KNN', 'RF', 'XGBoost', 'DNN'], price_bounds=(120000, 380000), step=20000, rows=2, figsize=(20,10),)
plt.show()
```


    
![png](./img/output_122_0.png)
    


For particularly expensive housing our model has significant loss, reaching hundreds of thousands of dollars for very expensive housing.


```python
models = ['LinReg', 'KNN', 'RF', 'XGBoost', 'DNN']
num_plots = len(models)
num_rows = 2
num_cols = (num_plots + 1) // num_rows

max_dim = 2e5

fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))

sub_200k_df = pred_df[pred_df['Price'] <= max_dim]

for idx, model_name in enumerate(models):
    row_idx = idx // num_cols
    col_idx = idx % num_cols
    axes[row_idx, col_idx].scatter(sub_200k_df['Price'], sub_200k_df[model_name], alpha=0.3)
    reg_line = sns.regplot(x=sub_200k_df['Price'], y=sub_200k_df[model_name], scatter=False, line_kws={'color': 'red'}, ax=axes[row_idx, col_idx])
    axes[row_idx, col_idx].set_xlabel('Actual Prices')
    axes[row_idx, col_idx].set_ylabel('Predicted Prices')
    axes[row_idx, col_idx].set_title(f'Actual vs Predicted Prices: {model_name}')
    axes[row_idx, col_idx].set_xlim([0, max_dim])
    axes[row_idx, col_idx].set_ylim([0, max_dim])
    real_line = axes[row_idx, col_idx].axline((max_dim, max_dim), slope=1, color='green')
    axes[row_idx, col_idx].legend(labels=[None, 'Predicted', None, 'Ideal'])
plt.tight_layout()
plt.show()
```


    
![png](./img/output_124_0.png)
    


This analysis reveals a few important trends:
1. Across the board, our models predict best on **cheaper housing** (i.e. &lt; \$120k) since that's where we have the most data.
2. All of our models &mdash; though each with very different learning paradigms and methodology &mdash; perform best on the same data and struggle with the same data.
   - All of our models also seem to (in general) underestimate value for housing greater than \\$50k and overestimate value for housing less than $50k

# Discussion

### Interpreting the result

A clear observation is that all of our models perform best on cheaper/standard-price housing. When applied to the real world, our model would be the most useful for the average consumer looking for an afforable place in Moscow. This result was expected, since most of our samples were for real estate with lower value, so of course our model performed better for these data.

Our main conclusion is that **Moscow real estate price predictions are limited primarly by data quality (feature richness)**. The Russian real estate market is a large, emerging market with houses going up for sale; our dataset reflects this, as it has many, many samples. However, interestingly, all of our models seem to perform similarly on similar price ranges. Linear regression, KNNs, random forests, XGBoost, and DNNs each have different strengths and weaknesses, and each have their own error cases. So, if the high error rates for less-expensive and more-expensive housing were due to learning methodology, then each of our trained models would likely show some variability in performance across the price ditribution. However, they do not, which means it's likely the case that our data is limited in feature granularity.

Our first subpoint of justification is examination of our learning curves from the model fitting section. Although our dataset is very large, thanks to CuML acceleration we were able to perfrom 3-fold CV for each model (with exception of the DNN, for which we could only estimate 'best' hyperparams by running on a limited number of models). Each classical ML model's learning curve inidicated that, while more data could marginally improve performance, it would not significantly change the loss. While it is not feasible to get a learning curve for the DNN, given the overfitting after only 100 epochs on a 100k-large sample size, it is unlikely that more data would improve performance significantly enough to fix the issues described above.

Our second subpoint of justification is that all of our models over/under-value housing in the same ranges in similar ways. While our best-performing models (Random Forest, XGBoost, DNN) are able to marginally improve results compared to KNN and Linear Regression, they still sucumb to price overesitmation of low-end housing and price underestimation of high-end housing. Our random forest and XGBoost models were built on trees, meaning they could learn a piecewise linear descision boundary, and DNNs can universally approximate any function. However, while these three models were our best-performing, they did not drastically outperform our linear regression. In reality, housing prices can vary due to several factors (style, location, proximity to public services and shopping centers, etc). This again indicates that our features are likely too simple, and additional features imposing some sort of non-linearity are necessary for more effective prediction.

Our final subpoint of note is that, if used in the real world given the data currently available, our models would be best used to predict low-end housing. Firstly, there is much less data available for high-end housing due to the low demand for such housing from the general public, so it is difficult to fit a model wihtout overfitting to the available data; we see this reflected in the very high MAE for housing worth more than $120k. Secondly, in a practical sense, high-end housing is likely even more subject to variation from metrics that are difficult to track (luxury features, 'art' of design and layout, etc.). These results fit well with our overall goal of making real estate value prediction more accessible and straightforward, since those selling expensive properties likely have resources for huamn prediction from professional agents.

### Limitations

Our findings indicate an important need for improved data in Moscow/Russian real estate markets. Because the real estate market in Russia (and, in our case, Moscow specfically) is emerging, data collection is likely not to the same standard as in places like America, where established housing and real estate services like Zillow have been collecting data on houses for years. To perform better predictive analysis on Moscow real estate prices would require sophisticated data centralization and colletion initiatives which have not been implemented yet. Once the market has matured more, and as more feature-rich data becomes available, ML and DL methods can be more easily applied for sophisticated predictive analysis. on many different ranges of pricing and many different types of real estate.

Additionally, our data does not include new costs from 2022 or 2023. Geopolitical and economic tensions in Russia have likely thrown markets like real estate, which is centered around long-term investment, into flux, so our model would likely incorrectly estimate prices for new housing available for the forseeable future as the Russian economy make long-term investments tumultuous.

Finally, additional hyperparam tuning for XGBoost and more complex DNN strucutres (e.g. very deep NN with skip connections, additional tuning for LeakyReLU, AdamW, etc hyperparams, etc) might result in better performance from these models. However, this sort of hyperparam tuning would require either better computational resources or more time.

### Ethics & Privacy

- The Russian economy is currently in a volatile position due to the war in Ukraine. If our model were to be used as a source of truth, and if it were too optimistic or pessimistic, we could wrongfully inflate the market or cause people to sell their homes for less than they are truly worth. Real estate investments can make or break one’s livelihood, especially in a turbulent and growing market like Russia, so making sure our model is functional and usable is important.
- The dataset doesn’t contain explicit personal information, but it contains information like date and time of listing publication and longitude/latitude location, which could potentially be used to identify individuals.
- The data is collected under specific legal provisions, which means it is collected lawfully, but it should be ensured that the use of this data for a machine learning project aligns with the original purpose of data collection.
- Any dataset has a potential for systematic biases, which could result in biased outcomes in a machine learning project. It is important to be aware of this and to either adjust the dataset to more fairly represent different groups or adjust the machine learning model to reduce bias in its prediction.

### Conclusion

Our exploration provides a usable model for price prediction on Moscow real estate, as well as justification for increased data-gathering initiatives to get more fine-grained, feature-rich data in emerging Russian real estate markets. Most famous western housing initiatives have been gathering large stores of data on which to build more complicated models, so higher quality data collection in the Russian real estate industry will likely result in better prediction for housing in 'standard' pricing ranges, as well as better prediction for housing in extreme price ranges.

Future work could also involve time-series analysis or prediction on other large Russian cities to determine if the above trends found in Moscow are consisten in other parts of Russia.

# Footnotes
<a name = "nahb"></a>1.[^](#nahb): Housing’s Contribution to Gross Domestic Product. https://www.nahb.org/news-and-economics/housing-economics/housings-economic-impact/housings-contribution-to-gross-domestic-product#:~:text=Share%3A,homes%2C%20and%20brokers'%20fees.<br> 
<a name="keyfactors"></a>2.[^](#keyfactors): Key Factors That Drive the Real Estate Market. https://www.investopedia.com/articles/mortages-real-estate/11/factors-affecting-real-estate-market.asp <br>
<a name="Redfin"></a>3.[^](#demographics): Is It Cheaper to Live in the City or the Suburbs?
. https://www.apartmenttherapy.com/suburbs-vs-city-cost-of-living-265646  <br>
<a name="Zestimate"></a>4.[^](#Zestimate): Building the Neural Zestimate
. https://www.zillow.com/tech/building-the-neural-zestimate/ <br>
<a name="Redfin"></a>5.[^](#Redfin): Redfin Estimate. https://www.redfin.com/redfin-estimate <br>
