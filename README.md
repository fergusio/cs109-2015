All right, this git repo is me working through [Harvard's 2015 CS109 class](http://cs109.github.io/2015/), one lab at a time.

The stuff to watch and work through:
- [Lecture videos & notes](http://cs109.github.io/2015/pages/videos.html)

Note: download the videos using [this script](https://github.com/christopher-beckham/cs109-dl-videos), and merge [pull request 11](https://github.com/christopher-beckham/cs109-dl-videos/pull/11) to get the 2015 videos.

My notes are here in the readme file as I work through the [course syllabus](http://cs109.github.io/2015/pages/schedule.html) and the [labs and homeworks](https://porter.io/github.com/cs109/content). 

**Study Suggestions before starting:**

- Use a internet blocker app [SelfControl](https://selfcontrolapp.com/) to stop procrastinating and a [pomodoro app] to break up study into chunks. 
- Buy a paper notebook and write notes as you watch the videos and do the labs.
- Get a second monitor so you can watch videos/have lab notebooks open and work through at the same time.
- Don't look at the lab and hw answers - try to do them first on your own. Discuss with others before looking at solutions.

## Week 1: What is Data Science

[Lecture 1](https://github.com/khalido/cs109-2015/blob/master/Lectures/01-Introduction.pdf):

- introduces data science.  
 

## Week 2: Intro Data Analysis and Viz

The lecture 2 notebook goes through getting data and putting it into a pandas dataframe.

Lab 1 has three very introductory notebooks: [pythonpandas](https://github.com/khalido/cs109-2015/blob/master/Labs/2015lab1/Lab1-pythonpandas.ipynb), followed by [babypython](https://github.com/khalido/cs109-2015/blob/master/Labs/2015lab1/Lab1-babypython.ipynb), and finally [git](https://github.com/khalido/cs109-2015/blob/master/Labs/2015lab1/Lab1-git.ipynb). However, since the course dates back to 2015, some of the python is a bit dated and uses 2.x code. 

After doing the three intro notebooks, [hw0](https://github.com/khalido/cs109-2015/blob/master/Labs/2015lab1/hw0.ipynb) runs you through installing anaconda, git, and setting up a github and aws account.

Hw0 has one interesting section, where you solve the [montyhall problem](https://en.wikipedia.org/wiki/Monty_Hall_problem) one step at a time. I didn't really like their steps, so made a [simpler monty hall implementation](https://github.com/khalido/algorithims/blob/master/monty_hall.ipynb).

Moving on to the [Lecture 2](https://github.com/khalido/cs109-2015/blob/master/Lectures/02-DataScraping.ipynb) & its [quiz notebook](https://github.com/khalido/cs109-2015/blob/master/Lectures/02-DataScrapingQuizzes.ipynb), this goes through some more pandas and data scraping web pages and parsing them.

I made a couple of notebooks expand on some of the stuff covered:

- [movielens notebook for basic pandas](https://github.com/khalido/cs109-2015/blob/master/movielens.ipynb) workflow of downloading a zip file, extracting it and putting into pandas dataframes and doing some q&a
- [twitter notebook](https://github.com/khalido/cs109-2015/blob/master/twitter.ipynb) - basic usage of twitter api and doing something with tweets 

**Lecture 3** ([slides](https://github.com/khalido/cs109-2015/blob/master/Lectures/03-EDA.pdf), [video](https://matterhorn.dce.harvard.edu/engage/player/watch.html?id=a4e81697-fd86-415c-9b29-c14ea7ec15f2)):

 - ask a q, get data to answer it, explore & check data, then model it and finally communicate and visualize the results.
 - keep viz simple and think of the [kind of chart](http://extremepresentation.typepad.com/blog/files/choosing_a_good_chart.pdf) needed. 


## Week 3 : Databases, SQL and more Pandas

[Lab 2](https://github.com/khalido/cs109-2015/blob/master/Labs/2015lab2/Lab2.ipynb) introduces web scraping with [requests](http://docs.python-requests.org/en/master/) and then parsing html with [beautiful soup 4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/.

Lecture 4 ([video](https://matterhorn.dce.harvard.edu/engage/player/watch.html?id=f8a832cb-56e7-401b-b485-aec3c9928069), [slides](https://github.com/cs109/2015/raw/master/Lectures/04-PandasSQL.pdf)) (covers some more [Pandas and SQL](https://github.com/khalido/cs109-2015/blob/master/Lectures/Lecture4/PandasAndSQL.ipynb).

Lecture 5 ([slides](https://github.com/cs109/2015/raw/master/Lectures/05-StatisticalModels.pdf), [video](https://matterhorn.dce.harvard.edu/engage/player/watch.html?id=873964c6-d345-4f46-a8bc-727b96432d63)) on stats is a bit sparse. Some supplementary material:
- [Stanford Statistics Course](https://lagunita.stanford.edu/courses/course-v1:OLI+ProbStat+Open_Jan2017/info) - check on this one vs the MIT one.
 - [Think Stats](http://greenteapress.com/thinkstats2/index.html) is a good basic book covering stats using Python.
 - [Think Bayes](http://greenteapress.com/wp/think-bayes/) follows on from Think Stats and covers Bayesian stats in Python.

## Week 4: Probablity, Regression and 
  
[Lab 3](https://github.com/khalido/cs109-2015/tree/master/Labs/2015lab3) has three notebooks:
- [Lab3-Probability](https://github.com/khalido/cs109-2015/blob/master/Labs/2015lab3/Lab3-probability.ipynb) covers basic probability. Uses a lot of numpy methods, so its a good idea to brush up on numpy.
    - [scipy.stats](https://docs.scipy.org/doc/scipy/reference/stats.html) - very handy, has most stats stuff needed for DS built in.
- [Lab3-Frequentism, Samples and the Bootstrap](https://github.com/khalido/cs109-2015/blob/master/Labs/2015lab3/Lab3-Freq.ipynb)
    - use seaborn for plotting, very handy. a [good guide to sns factorplot and facetgrids](http://blog.insightdatalabs.com/advanced-functionality-in-seaborn/)
    - [PDF](https://en.wikipedia.org/wiki/Probability_density_function) tells us the probability of where a continuus random variable will be in set of possible values that random variable can be (the sample space).
    - [PMF](https://en.wikipedia.org/wiki/Probability_mass_function) tells us the probability that a discrete random variable will be ecactly equal to some value
    - [CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function) function tells us the probability that a random discrete or continous variable X will take a value less than or equal to X. [Video](https://youtu.be/bGS19PxlGC4?list=PLF8E9E4FDAAA8018A)

### Lecture 6 ([slides](https://github.com/cs109/2015/raw/master/Lectures/06-StoryTelling.pdf), [video](https://matterhorn.dce.harvard.edu/engage/player/watch.html?id=afe70053-b8b7-43d3-9c2f-f482f479baf7)) is on how to tell stories with data. 

Good insights on how to tell a story with data. Infer, model, use an algorithim and draw conclusions (and check!).  

- Start with two fundamental questions:
    - Whats the goal? think first of that rather than going first to all the many ways you can slice and dice data.
    - Who cares? Know your audience and tell them a story. Have a clear sense of direction and logic. 
- Read some howto's on scientific writing
- have some memorable examples or small stories

Tell a story:
- know your audience and why/what they care about this data - what do they want?
- Don't make them think - clearly tell them what you want them to know in a way they can follow. highlight key facts and insights. 
- unexpectedness - show something the audience didn't expect. I liked the story which highlighted that bigfoot sightings are dropping sharply 
- What tools can we give the audience? For example, a web app for them to further explore the data, or a takeaway presentation with key points.
- be careful of your point of view and don't distort the data, but depending on the audience you can frame your story - for example presenting war deaths in red rather than a regular plot color.
- important to put the message up front - what does my graph show? Show it in stages if a lot of data, highlighting what to look at. Design matters. 

More resources:
- [The Functional Art](http://www.thefunctionalart.com/)

### Lecture 7 ([slides](https://github.com/cs109/2015/raw/master/Lectures/07-BiasAndRegression.pdf), [video](https://matterhorn.dce.harvard.edu/engage/player/watch.html?id=afe70053-b8b7-43d3-9c2f-f482f479baf7))

- think about bias, missing data, etc
- combine independent, unbiased estimators for a parameter into one:
    - fisher weighting
    - nate silver weighting method
- Bonferroni
- good segment on weighting variables
- regression towards the mean
- think of regression in terms of population or a projection of the column space of x - i.e what combination of the variables of x gets us closest to the value of y?
- linear regression means we're taking linear combination of predictors, the actual regression equation can be nonlinear
- what function of x gives the best predictor of y? 
- Gauss-Markov Theorem
- the residuals are the diff  b/w the actual value of y and the predicted value - plot residuals vs fitted values and vs each predictor variable - good way to eyeball quality of linear regression model
- variance R^2 measures goodness of fit, but doesn't mean model is good. 
- Best way to check a model is prediction.

## Week 5: Scikit learn & regression

### Lab 4 - Regression in Python ([video](https://matterhorn.dce.harvard.edu/engage/player/watch.html?id=483c8b93-3700-4ee8-80ed-aad7f3da7ac2), [notebook](https://github.com/khalido/cs109-2015/blob/master/Labs/2015lab4/Lab4-stats.ipynb))

- [Wikipeda article](https://en.wikipedia.org/wiki/Linear_regression)
- We have data X, which is related to Y in some way.
- Linear regression uses X to predict Y, and also tells us the predictive power of each variable of X
- Linear regression assumes the distribution of each Y is normal (which isn't always so)
- there are many ways to fit a linear regression model, most common is the [least squares](http://en.wikipedia.org/wiki/Least_squares) method
- be careful that the features (variables in X) aren't too similar
- explore your data, plot variables, scatterplots etc. Use seaborn to plot regression.
- use [sklearn to split dataset into a train/test](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
- use [cross-validation](http://scikit-learn.org/stable/modules/cross_validation.html) 
- overfitting happens when the model 'learns' the train data so performs better on that than the test dataset
- there are [many types of regressions](http://www.datasciencecentral.com/profiles/blogs/10-types-of-regressions-which-one-to-use) so think about which one to use
- for high d data, closed form vs gradient decent:
    - closed form - use math to solve. this becomes computationally intensive very quickly, is ordered n cubed
    - gradient descent is O(n), so for large high d data it's gradient descent all the way
- Logistic regression - used where outcome is binary, for example a chance of success/failure. read:[Adit's explanation.](http://adit.io/posts/2016-03-13-Logistic-Regression.html)

### Lecture 8: More Regression ([video](https://matterhorn.dce.harvard.edu/engage/player/watch.html?id=664f668e-e008-4f44-8600-e09ee6d629b0), [slides](https://github.com/khalido/cs109-2015/blob/master/Lectures/08-RegressionContinued.pdf))

- collinerarity - when some variables are highly correlated with each other - this is bad
- Logistic Regression
- Odds Ratio: ratio of two different people's odds of an outcome.
- Crude Odds Ratio Estimate - quick estimate but flawed as doesn't control for anything.
- Confounding Factors - i.e is one group pre-disposed to our outcome for some reason?
- Curse of dimensionality - in high d settings, vast majority of data is near boundaries, not center. But, high d can also be a [blessing](http://andrewgelman.com/2004/10/27/the_blessing_of/).
- dealing with high dimensionality: ridge regression, shrinkage estimation
- Stein's Paradox [wikipedia](https://en.wikipedia.org/wiki/Stein%27s_example), [good article](http://statweb.stanford.edu/~ckirby/brad/other/Article1977.pdf)
- LASSO and Ridge help with high D data by reducing features
    - Lasso does L1 regularization, reduces number of features
    - Ridge does L2 regularization, doesn't necessarily reduce features but reduces the impace of features on the model by reducing coefficient value
- Elasticnet does both L1 and L2 regularization

### Lecture 9: Classification ([video](https://matterhorn.dce.harvard.edu/engage/player/watch.html?id=c322c0d5-9cf9-4deb-b59f-d6741064ba8a), [slides](https://github.com/khalido/cs109-2015/blob/master/Lectures/09-ClassificationPCA.pdf))

- we take data and assign labels
- 1 nearest neighbour - simple classification method for low d data
    - slow, has to check all points to find nearest neighbour
- k nearest neighbours - use k nearest points to find decision boundary
    - find ideal k 
    - what distance function to use? 
    - my own very simple [kNN algo implementation](https://github.com/khalido/algorithims/blob/master/k%20nearest%20neighbours.ipynb)
- cross validation - for 5 fold cross validation, the data is split into 6 folds - 4 for training, one for validation and the sixth for testing, which is only used at the end.
- CIFAR-10 for 60K images - is split into 50K training and 10K test
    - pics are 32x32x3
- L1 distance is the absolute diff b/w two vectors
- L2 is the Euclidean distance i.e "ordinary" straight-line distance
- for images, l1 and l2 are pretty bad, so there are a lot more methods
- more features are good for classification, but too many features means the data gets sparse - the curse of dimensionality strikes
- so often we want to reduce dimensionality
- Principal Component Analysis - project a dataset from many variables into fewer less correlated ones, called the principal components. 
- Singular Value Decomposition (SVD) - computational method to calculate pricipal components of a dataset. It transforms a large matrix of data into three smallter matrixes: `A (m*n) = U(m*r) x E(r*r) x V(r*n)`. The values in the middle matrix `r*r` are the *singular* values and we can discard bits of them to reduce the amount of data to a more manageable number. 
- [good pca and svd explanation](https://medium.com/machine-learning-for-humans/unsupervised-learning-f45587588294)
- Watch [Statistics for Hackers](https://www.youtube.com/watch?v=Iq9DzN6mvYA)

### HW2 Q1 

- [notebook](https://github.com/khalido/cs109-2015/blob/master/homework/HW2.ipynb)
- Uses svd and pca to analyze gene data

## Week 6: SVM, trees and forests

Now the course finally gets interesting. Before starting this weeks work, think about project ideas and see [Hans Rosling](https://www.gapminder.org/videos/) videos to see how to present data. Pitch this project idea (to study group or the internet at large).

There are quite a few companies automating the entire datascience chain, so the key is being able to present your findings well.

### Lab 5: Machine Learning

**Learning Models** ([notebook](https://github.com/khalido/cs109-2015/blob/master/Labs/2015lab5/LearningModels.ipynb), [video](https://matterhorn.dce.harvard.edu/engage/player/watch.html?id=e509f996-9633-4b75-a48a-e29246a316db))

- we often have a small sample of a much dataset, and we want to predict the larger data from our sample.
- this isn't just statistical analysis, as we make models which involve domain knowledge and choices.
- need to think about whether our sample is in some way representative of the population
- Stochastic noise, i.e randomness
- systematic error, i.e where the sampling isn't representative, like polling ppl using landlines
- overfitting: models can 'memrize the the data points in the training set, becoming useless or inaccurate at predicting real world data. With many data sets a more and more complex dataset will keep getting better while getting worse on test/validation data. The best model minimizes test set error, and not training set error.
- great illustration of variance at 24:30 and 35min in the video
- use `from sklearn.cross_validation import train_test_split` for splitting datasets into a train test split. See [sklearn](http://scikit-learn.org/stable/)
- sklearn has 3 main features:
    - build and fit models
    - predict
    - transform data.
- sklearn expects days in a 2d array or a matrix of size `[n_samples, n_features]`, so reshape 1d data using `np.reshape(-1,1)`
- Validation - keep a chunk of data seperate to check the model after training on the test/train data.
- Cross Validation: randomly split data into K diff train/test splits - so you traion on K-1 partitions and test on 1, so there are a total of K combinations, leading to K risks. This leads to better results then just doing one test/train split. 
- regularization helps with overfitting

**Classification** ([notebook](https://github.com/khalido/cs109-2015/blob/master/Labs/2015lab5/Classification.ipynb), [video](https://matterhorn.dce.harvard.edu/engage/player/watch.html?id=90e73c64-855c-4b06-afb2-94da608ecfbf))

- sort data into classes, i.e what kind of fruit
- most datasets can be projected onto a lower dimensial space, for e.g using PCA
- read sklearn's PCA docs
- kNN:
- Logistic Regression - use [sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html), main paramater is C which defines how much to regularize the data. Read [this explanation](http://adit.io/posts/2016-03-13-Logistic-Regression.html)
- Use sklearns [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) to find hyperparameters
- one way to classify: use [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) to reduce the feature space, then use logistic regression to classify 
- many datatypes, like images, have tons of features, so important to reduce dimensionality.
- [sklearn PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) returns the principal components in order of how much they explain the data:

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=60) # PCA with no. of components to return
X = pca.fit_transform(data)
print(pca.explained_variance_ratio_) # how much of variance is explained
```

- sklearn uses the same interface for all its classifiers, so makes it easy to put a wrapper around gridsearchCV and pass in diff classifiers to compare.
- discriminative classifier - finds the decision boundary b/w classes
- maximum-margin classifier - for many classifincation problems, multiplie diff lines can seperate classes. Choose that line where the margin b/w classes is the largest, which makes this model senstive to boundaries b/w classes, rather than to point samples deep inside a class. 
- SVM is a discrimant classier which finds the widest possible margin b/w classes, including some points touching the boundary which are called the support vectors. (since they support or establish the margins.)


### Lecture 10: SVM, Evaluation

- KNN - training is fast, prediction slow since we need to check all the data points to find the nearest neighbours
- but if we know the decision boundary (the seperating hyperplane) we don't need all the data points
    - w: weight vector defines the orientation of the hyperplane, and bias b. 
    - so a new point x is classified by `w(transpose)*x + b`
    - this is the mathematical model of a neuron, invented 1957 by Rosenblatt
- step function vs sigmoid activation
- Support Vector Machines (SVM) are widely used, some consider it best of the shelf classifier. They add a new dimension to help seperate classes and also use maximum margin classification. SVM is called svm becuase of the support vectors defining the max margin lines for the classification boundary.
- large data is good for training svm as the points on the boundary are rare and svm cares about establishing the boundary
- since outliers can change the svm boundaries, there is a concept of slack variables - it allows the SVM to missclassify outliers to make a neat decision boundary. sklearn uses the parameter C to define the slack. the lower the number the more the slack.
- kernel tricks for svm - go to aribitarily mary dimensions with little computational cost. need to think about what kernel to use.
- read [Andrew Ng's cs229 svm notes](http://cs229.stanford.edu/notes/cs229-notes3.pdf)
- todo: tale sklearns 'faces' dataset and use svm to predict


### Lecture 11: Decision Trees and Random Forests

### HW 2 Questions 2,3 & 4 [notebook](https://github.com/khalido/cs109-2015/blob/master/homework/HW2.ipynb)




## Week 7: Machine Learning best practices

Around Week 7 or 8, start [project](http://cs109.github.io/2015/pages/projects.html).

> Towards the end of the course you will work on a month-long data science project. The goal of the project is to go through the complete data science process to answer questions you have about some topic of your own choosing. You will acquire the data, design your visualizations, run statistical analysis, and communicate the results. You will work closely with other classmates in a 3-4 person project team.

hw 3

## Week 8: EC2 and Spark

hw4

## Week 9: Bayes!

## Week 10: Text

hw5

## Week 11: Clustering!

## Week 12: Deep LEarning

## Week 13: Final Project & Wrapup


# Additional Resources
Stuff I found useful to understand the class material better.

- [Computational and Inferential Thinking](https://ds8.gitbooks.io/textbook/content/) - the textbook for UC Berkely's [Foundations of Data Science class](http://data8.org/.
- [Pythonb Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)
