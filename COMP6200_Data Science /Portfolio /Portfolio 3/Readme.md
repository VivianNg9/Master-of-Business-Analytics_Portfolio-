# Portfolio 3
## Analysis of an E-commerce Dataset Part 3 
In this portfolio, I continue to work on the modified cleaned E-commence dataset. 
The difference from those used in Portfolio 2 is that the ratings have been converted to like (with score 1) and dislike (with score 0).

 ### Description of Fields
* __userId__ - the user's id
* __timestamp__ - the timestamp indicating when the user rated the shopping item
* __review__ - the user's review comments of the item
* __item__ - the name of the item
* __rating__ - the user like or dislike the item
* __helpfulness__ - average rating from other users on whether the review comment is helpful. 6-helpful, 0-not helpful. 
* __gender__ - the gender of the user, F- female, M-male
* __category__ - the category of the shopping item

### The task sections
- Explore the dataset
- Convert object features into digit features
- Correlation between features 
- Logistic Regression Model 
- KNN Model 
- Tune the hyper-parameter K in KNN

### Outcome of the task 
- By performing hyperparameter tuning, the KNN model is optimized, and it seems to perform better with n_neighbors = 22.
- With an accuracy of approximately 74.5%, the tuned KNN model demonstrates the impact of hyper-parameter K tuning on model performance.
