# Check .ipynb file for codes and full project

## Abstract

Real estate is the foundation for many life milestones like owning a home, starting a family or a business, and more. However, it may be hard to break into the real estate world without first doing a lot of research and planning because real estate is after all, an investment. By building a ML model that helps predict the price of home, this can hopefully help to make the process easier for prospective homeowners and sellers. The data we will be using encompasses more than 5 million Russian real estate sales from 2018 - 2021 and has multiple variables such as the price of a house, listing date and time, geographic location, region, and information about the building (type, storeys, floor, living rooms, rooms). Although our dataset is in the Russian market, it provides us a lot of data points that can allow us to learn more about the different models and generalize it to different markets.

We will perform some EDA analysis to view the correlation of the different factors, and then build a linear regression model using CART regression, logistic regression, and random forest. We will then evaluate the performance of our model using mean absolute error (MAE).

## Background

The real estate market has been a pivotal factor and contributor in the economy as, according to the National Association of Home Builders, housing’s combined contribution to gross domestic product (GDP) generally averages 15-18%[1]. This percentage is calculated based on both residential investment and consumption spending on housing. Not only is housing a contributor to the economy but it is also an important asset to people’s lives as it not only signifies having a place to sleep in but is often perceived as a way to show one’s social status and a valuable asset where money can be allocated.

Despite the importance and high contribution that the market is to the economy, it has many factors that can quickly change the market. Although different factors can influence the real estate market, one of the most important factors is demographics[2].

Demographics consists of the data of the population near the location which affects the pricing of the house and also the demand the property has. Places in and near a major city could be more expensive due the proportion of square footage and price,[3] since major cities usually have limited land to be developed or already has no more space for new developments. Alhtough real estate predictions across different areas (urban, rural, suburban) can vary due to differences in land use, when considering a single location (e.g. one city) where land use, housing supply, etc., are more similar, demographics prevail as the key difference when comparing real estate.

Our exploration will, therefore, focus on Moscow, the capital of Russia which has a quickly-growing real estate market. Because the market is developing, it can be difficult for an average person to determine which variables contribute most to real estate pricing. By building a model to predict real estate pricing (in rubles), we aim to make distilling this demographic information easier on a larger scale.

There has been a lot of prior (and ongoing) research within the real estate industry, especially real estate companies such as Zillow with their “Neural Zestimate[4]," Redfin with their “Redfin Estimate[5]," and many other real estate companies also have their own models for estimating home prices. Since each model is built differently, this leads to varying price estimations. However, the bases of the models are similar as they take in large amounts of previous transactions and/or MLS data to get various variables to find good features to base the model off of as they keep retraining to get better results.

## Problem Statement

The real estate market can be a turbulent and rapidly changing environment, where it is often hard to predict the actual value due to many factors.Due to the multitude of different constants, we will focus our model on the general description of the property. We aim to make it easier for people to get this type of information by training a ML model on a large dataset of previous home purchases in order to predict what price point a home may be at.

## Data

Our current best candidate is the following dataset of Russian Real Estate pricing from 2018-2021. The dataset contains an incredible 5 million+ data points, with no null values and only a few thousand duplicate rows. Therefore, our data is very-well poised to avoid generalization without uses of techniques like cross-validation.

This massive dataset means training could takes many days or even weeks given our computational resources, which is not feasible. Since demographics data can vary between cities/counties. However, our exploration will primarily focus on Moscow. Thus, we are able to limit the size of our data to about 1/10th of the original dataset. Furthermore, if computational cost continues to be an issue, we may randomly sample a subset of our data to train (this will not harm any assumptions for the regression models we will use, since it does not violate any assumptions about the data which these models require)

There are 13 variables, 2 categorical, 2 location-based, and the rest ordinal. We will be removing the latitude and longitude columns as these prevent ethical issues regarding the location of homeowners and intense violations of privacy.

Each observation contains the price of a house, listing date and time, geographic location, region, and information about the building (type, storeys, floor, living rooms, rooms). Notably, it does not contain square footage, which is a landmark in much of the American real estate market.

Critical variables mostly encompass the house descriptions and the time of publishing. We will need to one-hot encode building type. Building type will not largely increase the width of the design matrix.

Finally, we will need to convert data and time of publication to only the year, and potentially also the month, in case we’d like to do time series analysis. As mentioned earlier, we’ll also remove the latitude and longitude due to concerns of privacy. Finally, for our non-tree models, we will also normalize our data points by z-score, since data like price in rubles will be orders of magnitude larger than number of rooms.

## Proposed Solution

Note that we discuss error metrics, including justifications for L1 loss (MAE), in the Evaluation Metrics section.

Before discussing our implementation, regarding benchmark models: there are some models available on Kaggle using time series analysis, which might result in good outputs. However, there are no significant authorities on Russian real estate pricing in machine learning, especially since this is an emerging market. Furthermore, American authorities on real estate prediction often keep their models internal as a part of their business model, so it is difficult to use existing robust benchmark models without APIs and the sort. First, it is important to note that our dataset is massive. With over 5 million samples, our model will certainly generalize well, but this also means we may have too many confounding variables and our model may not reach high enough MAE. During EDA, we will determine cities which contain interesting data, and we can fragment our data by city. Depending on computational resources and time constraints, we may choose multiple cities, or only use one.

Second, regardless of which or how many cities we use, this data is simply far too massive for any form of CV. Additionally, CV is not necessary here, since our validation set is likely to generalize well. Finally, luckily much of the data is ordinal, with few categorical variables with limited possible values. For our regression models, we may try to avoid extra data points such that we don’t have too many features in our design matrix. However, to attempt to include these features in at least one model, we will also try random forests.

CART Regression
Linear regression using L1 loss
After performing EDA, if certain metrics seem like they could use polynomial features, we can also try polynomial regression using L1 loss.
Random Forests to include categorical variables.
We can also try variants of linear and polynomial regression using L2 regularization. It is unlikely that many of these features will be confounding (though we can confirm with EDA), so L2 regularization is likely more reasonable. We can also try mixed regularization in case some features are, indeed, confounding.

Then, if we have enough computational resources, we can perform grid search on different hyperparameters for model selection. However, if this is not feasible, we can empirically justify pruning techniques, regularization mix, etc.

Finally, we will use sklearn for all implementations for 1) readable code, and 2) efficient, thoroughly tested implementations of the algorithms discussed above. While tools like Keras do have gpu acceleration, these methods aren’t as useful for our models as compared to neural network models.

## Evaluation Metrics

The three most common metrics for regression are mean squared error (MSE), mean absolute error (MAE) and root mean squared error (RMSE). MSE and RMSE heavily penalize outliers, while MAE proportionately penalizes all errors. Our data includes some more extreme outliers (10 living rooms, 39th floor, etc). For these ‘extreme’ sorts of houses, there are also many extra possible factors beyond measurable features like number of rooms; for example, the ‘art’ of designing expensive homes with luxury features. So, using MSE or RMSE would likely bias our model to these extreme outliers while lowering our model’s success in gauging prices for a majority of houses on the market. Conversely, MAE would result in a better representation of the data for a majority of ‘normal’ cases. Therefore, we will stick to MAE.

## Model Fitting
When we began training models, an immediate issue presented itself: we didn't have sufficient compuational resources to handle very large data using only Sklearn's cpu-only implenetaions. This made training very slow and hyperparameter tuning out of the question.

However, training only on our data with hyperparams chosen heurisitically did not offer good enough results. So, in order to allow for hyperparam tuning on our large data, we implement Nvidia's RAPIDS API, which offers models with similar syntax to Sklearn, but with GPU acceleration. In particular, RAPIDS includes the CuML package, which accelerates training significantly.

The main challenge with using RAPIDS was technical implementation: we acheived most stability on WSL2 Ubuntu on Windows with CUDA 11.5, and some functions in CuML seem less stable (i.e. more prone to crashing or hanging on computation) with this setup.

Finally, although we will be testing different models, we will not be peforming Nested CV for algorithm selection. Even with the RAPIDS API, performing Nested CV would be simply too computationally intensive, and also likely wouldn't realize better results on our large data.

## Discussion

### Interpreting the result

A clear observation is that all of our models perform best on cheaper/standard-price housing. When applied to the real world, our model would be the most useful for the average consumer looking for an afforable place in Moscow. This result was expected, since most of our samples were for real estate with lower value, so of course our model performed better for these data.

Our main conclusion is that Moscow real estate price predictions are limited primarly by data quality (feature richness). The Russian real estate market is a large, emerging market with houses going up for sale; our dataset reflects this, as it has many, many samples. However, interestingly, all of our models seem to perform similarly on similar price ranges. Linear regression, KNNs, random forests, XGBoost, and DNNs each have different strengths and weaknesses, and each have their own error cases. So, if the high error rates for less-expensive and more-expensive housing were due to learning methodology, then each of our trained models would likely show some variability in performance across the price ditribution. However, they do not, which means it's likely the case that our data is limited in feature granularity.

Our first subpoint of justification is examination of our learning curves from the model fitting section. Although our dataset is very large, thanks to CuML acceleration we were able to perfrom 3-fold CV for each model (with exception of the DNN, for which we could only estimate 'best' hyperparams by running on a limited number of models). Each classical ML model's learning curve inidicated that, while more data could marginally improve performance, it would not significantly change the loss. While it is not feasible to get a learning curve for the DNN, given the overfitting after only 100 epochs on a 100k-large sample size, it is unlikely that more data would improve performance significantly enough to fix the issues described above.

Our second subpoint of justification is that all of our models over/under-value housing in the same ranges in similar ways. While our best-performing models (Random Forest, XGBoost, DNN) are able to marginally improve results compared to KNN and Linear Regression, they still sucumb to price overesitmation of low-end housing and price underestimation of high-end housing. Our random forest and XGBoost models were built on trees, meaning they could learn a piecewise linear descision boundary, and DNNs can universally approximate any function. However, while these three models were our best-performing, they did not drastically outperform our linear regression. In reality, housing prices can vary due to several factors (style, location, proximity to public services and shopping centers, etc). This again indicates that our features are likely too simple, and additional features imposing some sort of non-linearity are necessary for more effective prediction.

Our final subpoint of note is that, if used in the real world given the data currently available, our models would be best used to predict low-end housing. Firstly, there is much less data available for high-end housing due to the low demand for such housing from the general public, so it is difficult to fit a model wihtout overfitting to the available data; we see this reflected in the very high MAE for housing worth more than $120k. Secondly, in a practical sense, high-end housing is likely even more subject to variation from metrics that are difficult to track (luxury features, 'art' of design and layout, etc.). These results fit well with our overall goal of making real estate value prediction more accessible and straightforward, since those selling expensive properties likely have resources for huamn prediction from professional agents.

## Limitations

Our findings indicate an important need for improved data in Moscow/Russian real estate markets. Because the real estate market in Russia (and, in our case, Moscow specfically) is emerging, data collection is likely not to the same standard as in places like America, where established housing and real estate services like Zillow have been collecting data on houses for years. To perform better predictive analysis on Moscow real estate prices would require sophisticated data centralization and colletion initiatives which have not been implemented yet. Once the market has matured more, and as more feature-rich data becomes available, ML and DL methods can be more easily applied for sophisticated predictive analysis. on many different ranges of pricing and many different types of real estate.

Additionally, our data does not include new costs from 2022 or 2023. Geopolitical and economic tensions in Russia have likely thrown markets like real estate, which is centered around long-term investment, into flux, so our model would likely incorrectly estimate prices for new housing available for the forseeable future as the Russian economy make long-term investments tumultuous.

Finally, additional hyperparam tuning for XGBoost and more complex DNN strucutres (e.g. very deep NN with skip connections, additional tuning for LeakyReLU, AdamW, etc hyperparams, etc) might result in better performance from these models. However, this sort of hyperparam tuning would require either better computational resources or more time.

## Ethics & Privacy

The Russian economy is currently in a volatile position due to the war in Ukraine. If our model were to be used as a source of truth, and if it were too optimistic or pessimistic, we could wrongfully inflate the market or cause people to sell their homes for less than they are truly worth. Real estate investments can make or break one’s livelihood, especially in a turbulent and growing market like Russia, so making sure our model is functional and usable is important.
The dataset doesn’t contain explicit personal information, but it contains information like date and time of listing publication and longitude/latitude location, which could potentially be used to identify individuals.
The data is collected under specific legal provisions, which means it is collected lawfully, but it should be ensured that the use of this data for a machine learning project aligns with the original purpose of data collection.
Any dataset has a potential for systematic biases, which could result in biased outcomes in a machine learning project. It is important to be aware of this and to either adjust the dataset to more fairly represent different groups or adjust the machine learning model to reduce bias in its prediction.
Conclusion

Our exploration provides a usable model for price prediction on Moscow real estate, as well as justification for increased data-gathering initiatives to get more fine-grained, feature-rich data in emerging Russian real estate markets. Most famous western housing initiatives have been gathering large stores of data on which to build more complicated models, so higher quality data collection in the Russian real estate industry will likely result in better prediction for housing in 'standard' pricing ranges, as well as better prediction for housing in extreme price ranges.

Future work could also involve time-series analysis or prediction on other large Russian cities to determine if the above trends found in Moscow are consisten in other parts of Russia.
