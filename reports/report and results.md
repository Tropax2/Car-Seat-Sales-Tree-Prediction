## The Dataset 

The dataset contains data information about car seat sales in 400 stores. The columns of the dataset are the following:

- **Sales**: Unit sales at each location (typically measured in thousands of units).
- **CompPrice**: Price charged by a competitor for a similar product (in dollars).
- **Income**: Average community income for the store’s area (commonly in thousands of dollars).
- **Advertising**: Local advertising budget for the company’s product (commonly in thousands of dollars).
- **Population**: Population size in the store’s area (commonly in thousands of people).
- **Price**: Price the company charges for the car seat at that location (in dollars).
- **ShelveLoc**: Quality of the shelf location in the store (Bad / Medium / Good).
- **Age**: Average age of people in the store’s area (in years).
- **Education**: Education level in the store’s area (typically years of education).
- **Urban**: Whether the store is in an urban location (Yes / No).
- **US**: Whether the store is in the United States (Yes / No).

The target variable is the `Sales` variable, while the others are predictor variables. 

## Data Processing 

The CSV file is loaded into a pandas DataFrame and rows with missing values are dropped. Numerical predictors are standardised using `StandardScaler`, while categorical predictors are one-hot encoded using `OneHotEncoder` (combined with `ColumnTransformer`). The preprocessing steps are fit on the training set and then applied to the test set to avoid data leakage.

## Models Used and Results Obtained  

### Regression Tree 

We first fit a regression tree model with a small tree and no pruning (cost-complexity-pruning parameter equal to 0) with `max_depth` equal to 3 as the maximum number of levels, which, in a validation set approach, that is, an 80/20 single train/test split, obtains a test MSE of 6.68. Next, we perform 5 fold cross-validation to determine the optimal `max_depth` parameter in order to achieve optimal level of tree complexity. The optimal value is 5, where we see a reduction of test MSE to 5.87.

Following this approach, we tested if pruning the tree would be worth it, and so, by performing cross-validation the optimal `max_depth` parameter changed from 5 to 6, and the cost-complexity-pruning parameter changed to 0.05. The test MSE obtained with this model is 5.28. Hence, pruning reduced the test MSE by ~10%, indicating a better generalization and that the depth only tree slightly overfits the data.

### Bagging 

We fit a bagging ensemble model of decision trees to the data and compute the test MSE. The model is fit with 500 trees and using all the 
predictors.

We first define the trees to have `max_depth` equal to 5 and with pruning parameter equal to 0, obtaining a test MSE of 3.5. We repeat the process, but with the optimal parameters found for a single decision tree, which causes the test MSE to drop to 3.44.

### Random Forest 

We fit a random forest ensemble model to the data and compute the test MSE. The model is fit with 500 trees and using all the predictors.

We first define the trees to have `max_depth` equal to 5 and with pruning parameter equal to 0 like it was done for bagging, obtaining a test MSE of 3.48, equal to bagging. We repeat the process, but with the optimal parameters found for a single decision tree, which causes the test MSE to drop to 3.45, basically the same as bagging.

This result is similar to bagging because we used all features in each split.

If we only use around the square root of all the predictors, that is, 4 variables then we get 4.14 test MSE for trees with `max_depth` equal to 5 and no pruning; and test MSE equal to 3.93 for trees with optimal parameters. This drop in performance may be explained by the fact that the 
predictors chosen are not the most significant (check paragraph below) and so individual splits are weak when few predictors dominate.

We also check which are the most significant predictors for the model and the top three are: Price, ShelveLoc_Good and Age.

### Boosting

We fit the boosting ensemble model to the data and compute the test MSE. The model is fit with 500 trees and a learning rate of 0.05.

We first define the trees to have `max_depth` equal to 5 and with pruning parameter equal to 0 like it was done for bagging, obtaining a test MSE of 2.72, so that the performance is superior to both bagging and random forest. 

## Conclusion

Across all models, ensemble methods substantially improve predictive performance over a single regression tree. Bagging achieved the best test performance (test MSE ≈ 3.44), while a random forest using all predictors at each split performed similarly (test MSE ≈ 3.45). Using the standard random-forest setting `max_features`≈√p (4 predictors) led to worse performance (test MSE ≈ 3.93), suggesting that a small number of strong predictors (notably Price, ShelveLoc_Good, and Age) drives most of the predictive power, and restricting candidate features at each split can weaken the quality of splits. Results are reported for a single 80/20 train/test split with hyperparameters selected via cross-validation on the training set; repeating the analysis across multiple random splits would provide a more robust estimate of generalization performance.