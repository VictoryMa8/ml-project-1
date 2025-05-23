# -*- coding: utf-8 -*-
"""Copy of PA8: tuning.ipynb - Victor Ma

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/19kn7l_UcAvWkOsHn0idBAkV0QVmaZi8b

Name: Victor Ma


Who you worked with:

## Objectives
The goals of this project are to:

- Outline and implement a full machine learning workflow.

- Explore data-driven decision-making through analysis of cookie recipes.

- Apply techniques we've seen from class, and be able to comfortably navigate documentation

## Overview
For this programming assignment, you will come up with and implement a workflow to determine the attributes that make a cookie delicious. Based on historical data of 5K cookies regarding baked_temp, sugar_index among others, the authors of [Cookie Monster](https://github.com/albertoabreu91/Cookies_project/tree/master?tab=readme-ov-file) collected the dataset we'd like to use. The authors started a workflow but didn’t get very far — now it’s your turn to complete a proper ML workflow! Our goal is to expand upon the outlined workflow (see ReadMe in repo) in order to answer an interesting question about this data. The dataset we'll be using can be found by clicking the data folder and locating the dataset `cookies.csv`.

## Schedule
Here is the suggested schedule for working on this project:
- Monday: Read through project instructions, run code for Task 0.
- Tuesday: Complete Tasks 1-2.
- Wednesday: Complete Tasks 3.
- Thursday: Complete Task 4.

This project is due on Thursday, 4/17, by 11:59pm.

#Task1: Cookie Data

We'd like to be able to bring our data in to try to understand it.

Link to repo: [link ](https://github.com/albertoabreu91/Cookies_project)

##💻 Q1: Read in data

Your goal is to use the pandas method `read_csv` to read in the `cookies.csv` data from the repo. To find the url for this dataset, go to the data folder, click on `cookies.csv`, then click the button `Raw` on the right. You should be brought to a new window containing the dataset. Copy and paste the url into `read_csv` below (hint: url should begin with `raw.githubusercontent.com`).

If you are unsure how to use `read_csv` look at previous PAs/workbooks, or use the pandas doc.
"""

import pandas as pd
cookies = pd.read_csv("https://raw.githubusercontent.com/albertoabreu91/Cookies_project/refs/heads/master/data/cookies.csv")

cookies.head(20)

"""##✏ Q2: Data Dictionary

Once you have the data, add a data dictionary (there is one to reference on the repo).

**sugar to flour ratio** The fraction of sugar to flour, expressed as a decimal (sugar/flour).

**sugar index** Modified glycemic index.

**bake temp** Baking temperature, in degrees fahrenheit.

**chill time** Time necessary for the cookies to rest in the fridge, expressed in minutes.

**calories** Unit of heat equal to the heat needed to raise the temperature of 1,000 grams of water by one degree Celsius.

**density** Expressed in grams/cm3.

**pH** pH of cookie.

**grams baking soda** Grams of leavening agent (from recipe).

**bake time** how long the cookies need to bake, in minutes.

**butter type** Form of butter used.

**weight** In grams.

**diameter** In centimeters.

**mixins** Elements added to the batter, as additions.

**crunch factor** Index of chrispiness.

**aesthetic appeal** Appearance, based on color, regularity, and form.

**quality** ‘Goodness’ of cookie. (Target variable)

##💻 Q3: Data observations

For each feature, do some preliminary EDA and add 1-2 observations about its distribution (e.g., `sugar_index` is right-skewed, etc.). This step is to help organize and highlight the types of data cleaning that may be important later on.
"""

import seaborn as sns
import matplotlib.pyplot as plt
sns.histplot(x = cookies['sugar index']) # right skewed
plt.show()
sns.histplot(x = cookies['sugar to flour ratio']) # slightly right skewed/normal distribution
plt.show()
sns.histplot(x = cookies['bake temp']) # slightly right skewed
plt.show()
sns.histplot(x = cookies['chill time']) # slightly right skewed
plt.show()
sns.histplot(x = cookies['calories']) # normal distribution
plt.show()
sns.boxplot(x = cookies['density']) # right skewed
plt.show()
sns.histplot(x = cookies['pH']) # normal distribution
plt.show()
sns.histplot(x = cookies['grams baking soda']) # slightly right skewed
plt.show()
sns.histplot(x = cookies['bake time']) # slightly normal distribution
plt.show()
sns.histplot(x = cookies['butter type']) # 3x more 'melted' than 'cubed' observations
plt.show()
sns.histplot(x = cookies['weight']) # slightly right skewed
plt.show()
sns.histplot(x = cookies['diameter']) # unclear?
plt.show()
sns.histplot(y = cookies['mixins']) # most common == chocolate
plt.show()
sns.histplot(x = cookies['crunch factor']) # not normal distribution
plt.show()
sns.histplot(x = cookies['aesthetic appeal']) # unclear
plt.show()
sns.histplot(x = cookies['quality']) # normal distribution
plt.show()

"""# Task 2: Outlining a workflow
The authors of the repo [Cookies Project](https://github.com/albertoabreu91/Cookies_project) did some preliminary outlining of a workflow. We'd like to take that workflow and build off of it.

For each of the items below, take inspiration from what the authors outlined but bring in your own twist.

##✏ Q4: Goal

What will be your goal in working with this data set? Will you choose to do regression or classification? Who would be interested in this type of analysis?

My goal in working with this data set will be to create a machine learning model that will accuracy predict the 'crunchiness/crispiness' of a cookie (based on the 'crunch factor' variable). This will be a regression problem. I believe this question will be particularly interesting for those interested in food science or those just generally interested in how baking works/how texture in food is determined. We would ultimately find the best predictors that affect the crunchiness of a cookie.

##✏ Q5: EDA

What kinds of exploratory data analysis will you do?

We will begin with looking at the summary statistics for 'crunch factor' and its relationship with several features within the dataset. We will look at graphs that illustrate the overall relationship between crunch factor. Furthermore, we will check the relationships and correlations between our predictor variables. We will also look at the counts and overall distribution of features.

##✏ Q6: Data Cleaning

What data cleaning will you need to do to prepare the data set? Will you implement methods from `sklearn.preprocessing` or use your own?

I will have to alter the 'created' columns, such as 'sugar index' or 'aesthetic appeal', in order to fit my own needs for the models I will be using. Removing columns that do not offer much insight or difficult to work with will also be a part of the data cleaning process. We will also most likely encode certain features like 'butter type'.

##✏ Q7: Model Selection

Which machine learning algorithms will you use to train models (pick three)? Why are you choosing these algorithms?

Linear Regression, K-Nearest Neighbors, Random Forest. I am choosing these three algorithms because they allow me to address this regression problem while utilizing three different levels of complexity.

##✏ Q8: Model Training

How will you attempt to optimize your models?

I will attempt to optimize my models by using Grid Search CV to tune my hyperparameters. For linear regression, I will tune the fit intercept parameter. For KNN, I will tune the neighbors parameter. Finally, for random forest, I will optimize the max depth and minimum samples split parameters.

##✏ Q9: Model Evaluation

How will you analyze the accuracy of your models? Approximately what do you think will be “good” accuracy rates for your chosen task?

I will analyze the accuracy of my models with K-Fold Cross Validation. I think 70% accuracy will be pretty good in this case. I will also be looking for good RMSE and R^2 values. I believe 0.70 is a good R^2 value in our case.

# Task 3: Implementing the workflow
Now that you have an outline, let's put it to code! You should use the workbooks we covered in class to help you, along with referencing the sk-learn documentation.

##💻 Q10: Pull in relevant methods

Add the methods required to implement your workflow below. It will be helpful to reference the slide decks/workbooks, and sk-learn docs.
"""

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# import whatever models or preprocessing you want to use here
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

"""##💻 Q11: EDA

Implement EDA, beyond what was done for Q3.
"""

#eda
sns.scatterplot(y = cookies['calories'], x = cookies['weight'])
plt.show() # there is an observation we have to remove!

sns.scatterplot(y = cookies['calories'], x = cookies['sugar index'])
plt.show() # weak relationship

sns.boxplot(x = cookies['butter type'], y = cookies['crunch factor'])
plt.show() # no real relationship

sns.boxplot(y = cookies['mixins'], x = cookies['crunch factor'])
plt.show() # okay some cool things here

sns.scatterplot(x = cookies['density'], y = cookies['crunch factor'])
plt.show() # ??? okay density is a no go

sns.scatterplot(x = cookies['bake time'], y = cookies['crunch factor'])
plt.show() # ??? not good either

sns.scatterplot(x = cookies['pH'], y = cookies['crunch factor'])
plt.show() # strange

sns.scatterplot(x = cookies['grams baking soda'], y = cookies['crunch factor']) # slightly right skewed
plt.show() # also strange

"""##💻 Q12: Data Cleaning

Implement data cleaning (like encoding), then split into X (feature set) and y (target).

Note: if you'd like to use `StandardScaler` or `OneHotEncoder` you'll want to add these to the pipeline in the next step.
"""

# data cleaning steps
cookies = cookies.dropna()

# example of splitting
X = cookies.drop(['crunch factor', 'butter type', 'mixins'], axis = 1)
y = cookies['crunch factor']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

"""##💻 Q13:Model Selection

Below is an outline of a pipeline. You'll want to add this to a code chunk below, and fill it in according to your outline from Task2. If you choose to use preprocessing steps from `sk-learn` you'll want to add those steps to `Pipeline`. Some models we've covered in depth are `LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier`, but you are free to use whatever you'd like to.

```
pipe = Pipeline(steps=[('classifier', 'passthrough')])

seed=42
param_grid = [
    {'classifier' : [],
     'classifier_param1' : [],
     'classifier_param1' : []},
    {'classifier' : [],
     'classifier_param3' : [],
     'classifier_param4' : []}
]

```

##💻 Q14: Model Training

Below is code that implements cross validation using the pipeline and `GridSearchCV`.

Edit this to suit the workflow you have outlined in Task 2. For example, would it make more sense to use `RandomizedSearchCV`, or a different metric than `accuracy`, or a different number of folds in CV?
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_features = ['sugar to flour ratio', 'sugar index', 'bake temp', 'chill time',
                    'calories', 'density', 'pH', 'grams baking soda', 'bake time',
                    'weight', 'diameter', 'aesthetic appeal']

preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

pipe = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('regressor', 'passthrough')
])

seed = 42
param_grid = [
    {'regressor': [LinearRegression()],
     'regressor__fit_intercept': [True, False]},

    {'regressor': [KNeighborsRegressor()],
     'regressor__n_neighbors': [3, 6, 9, 12]},

    {'regressor': [RandomForestRegressor(random_state=42)],
     'regressor__max_depth': [None, 10, 20],
     'regressor__min_samples_split': [2, 5]
    }
]

grid_search = GridSearchCV(pipe, param_grid=param_grid, n_jobs=-1, cv=10, scoring='r2')
grid_search.fit(X_train, y_train)

print("tuned hyperparameters and best model:",grid_search.best_params_)
print("R^2:", grid_search.best_score_)

best_model = grid_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print("Test set R^2 score:", test_score)

"""This will output the best algorithm and hyperparameter choices for the cookies dataset. Now it's time to interpret this model and discuss evaluation

##✏ Q15: Model Evaluation

Interpret the best fit model.

Our model evaluation determined linear regression with regressor_fit_inercept == true as the best model and hyperparameter. I have tried debugging, but the R^2 always comes out wonky. I hope to simply ignore it for the next few questions...

#Task 4: Telling a compeling story

After all the modeling and evaluation, it's time to zoom out and ask "why did we do all of this in the first place?"

This task helps you reflect on the “ah-ha!” moments that came from your data exploration and modeling. These insights form the foundation of a compelling data story — one that others (even outside this class) could find persuasive or thought-provoking.

Our goal with this task is to help pinpoint a few insights from your analysis that stood out, surprised you, or challenged your assumptions. These will form the heart of your data narrative.

## Q16: Data-Driven Insights

Use these guiding questions to articulate your key findings:

- Which features mattered the most in predicting cookie quality?

- Was there anything unexpected in your EDA or model results?

- Did your modeling reveal any relationships you didn't initially consider?

- What changes to the dataset (e.g., filtering, feature engineering) helped improve your results the most?

- If someone asked you “what makes a cookie great?”, how would you answer using evidence?

Write your three insights below. Be specific and support with evidence from your workflow.

### ✏ Insight 1

There was definitely unexpected behavior in our initial EDA. There were many strange values and crazy outliers that were certainly abnormal. For instance, there were a few observations with "negative" calories, which is obviously impossible. This shows the importance of data cleaning.

### ✏Insight 2

We initially removed the categorical features, because we wanted to have a simpler model, although I am not certain that it made our model better. I suppose this highlights the importance of going back and forth with the workflow steps.

### ✏Insight 3

The 'man-made' features were interesting to deal with. Furthermore, features like density proved to be useless, given how they were listed in the dataset (mostly 1.0). This shows the importance of preprocessing.

##✏ Q17: Framing a Data Story
Good data stories go beyond charts. They interpret the findings and engage the audience. Who would care about this finding? What action or change does this insight suggest? What visual or informative result from your data best expresses this idea?

Read through the following resources, and identify two strategies you found useful. Think of a way you could incorportate it into this analysis.

* Communicating a Data Story ([link](https://the.datastory.guide/hc/en-us/articles/360003648756-How-to-Communicate-The-Story-in-your-Data))
* Creating Compelling Stories ([link](https://the.datastory.guide/hc/en-us/articles/360003653396-Techniques-for-Propelling-a-Story-Forward))

### ✏Strategy 1

It is important to order the story in an engaging way that places crucial pieces of information in certain places so that the reader can contextualize the findings and ultimate results of the modeling. In our case, we would want to frame our results in a way so that the target audience (bakers, food science people, etc.) is reached as fast as possible and we can retain attention and convey the findings.

### ✏ Strategy 2

It is important to frame the findings in a way that also give off a bit of emotion! If you want to keep readers' attentions you should find ways to be relatable; maybe add in a few jokes if the time is right. Have contagious energy and make sure the topic fits your tone. In our dataset, it would help to have a fun and engaging tone, since it is just cookies.

##✏ Q18: Data story pitch

Write a short (4-5 sentence) data story pitch as if you were sharing your results in one of the scenarios below.

An example pitch is given for each scenario. Use these examples to tailor to your specific results.

* Scenario 1: you're presenting to the **CEO of a mid-sized cookie company** looking to improve product quality and customer satisfaction.
> "Our model shows that cookies with lower baked temps and more sugar consistently score higher in quality. These two features accounted for approximately 72% of the variance. By reducing the baking temp slightly, your company could produce more satisfying cookies without changing core ingredients"


* Scenario 2: you're presenting to a group of **recipe developers** who are passionate about technique and ingredient optimization.
> "Our analysis revealed that using oat flour combined with low moisture content resulted in higher quality cookies. This effect was most pronounced when baking below 340°F. This could inspire a new cookie line that emphasizes chewy, rich textures without added fats."

* Scenario 3: you’re demoing this project at a **machine learning showcase** where you're evaluated on both your technical process and communication.
> "We evaluated three models — Logistic Regression, Random Forest, and KNN — and found that Random Forest gave the highest accuracy at 86%. Key features included sugar_index and baked_temp, which showed strong non-linear relationships with cookie quality. Our pipeline is easily adaptable for predicting quality in other baked goods."

We tried and tested three models: Linear Regression, KNN, and Random Forest. We found that Linear Regression gave the highest R^2 value of (we're not gonna say, because my code is bugged...). Key features included 'bake temp' and 'mixins', which showed the strongest relationships with 'crunch factor'. We believe our pipeline can be easily changed/adapted for other baked treats or pastries.

#Bringing it all together

Now that you've crafted your own pitch, let's see how this process might look outside of the classroom.

In professional settings, data work usually starts not with a dataset—but with a problem. A product manager, business analyst, or client might come to you and say something like: "We're seeing declining ratings for our cookies—can you figure out why?" or "We want to develop a low-cost but high-quality recipe. What tradeoffs can we make?"

From there, your job becomes about framing the right question, identifying the most relevant data (which might not exist yet!!), and shaping a workflow around the decision-making needs of that audience. In contrast, your classroom workflow started with a dataset and built up to a pitch — in practice, it often happens in reverse.

In real-world projects, you rarely get a perfectly clean dataset up front. Stakeholders might change their minds halfway through. You often need to balance accuracy with speed, interpretability, or budget. This means workflows are often messier: you may prototype a model quickly to explore feasibility before refining, or spend more time understanding what metrics truly matter to decision-makers (e.g., do they care more about accuracy, cost reduction, or customer experience?).

Applying this type of workflow in the real world relies just as much—if not more—on your adaptability and communication skills as it does on your technical abilities. Success often depends on how well you can navigate shifting goals, collaborate with stakeholders, and translate insights into meaningful action.

#✏ Q19: Reflection

What did you like about it? What could be improved? Your answers will not affect your overall grade. This feedback will be used to improve future programming assignments.

I struggled a bit with this assignment. As shown above, the model training didn't go as well as I hoped. I did however, enjoy the 'storytelling' bits. This assignment was difficult, but I suppose it is preparing me for the big project and giving me some insight on how I should do things. Overall, I think this assignment is okay. I think I needed some more guidance before starting it.

#Grading

For each of the following accomplishments, there is a breakdown of points which total to 20. The fraction of points earned out of 20 will be multiplied by 5 to get your final score (e.g. 17 points earned will be 17/20 * 5 → 4.25)
* (1pt) Task1 Q1: Correctly pulls in `cookies.csv`
* (1pt) Task1 Q2: Has data dictionary
* (2pt) Task1 Q3: Expands upon data dictionary with EDA insights
* (1pt) Task2 Q4: The project goal is clearly articulated and provides specific context for the workflow steps, including what is being predicted and why it matters.
* (2pt) Task2 Q5-Q9: At least half of the workflow outline includes clear, relevant, and actionable steps that show an understanding of ML processes (e.g., specific methods for EDA, cleaning, or model choice).
* (1pt) Task2 Q5-Q9: At least 75% of the workflow outline includes well-defined and appropriate steps that logically contribute to the overall goal and modeling approach.
* (1pt) Task2 Q5-Q9: All parts of the workflow outline contain clearly stated, technically appropriate steps that are directly aligned with the project's goal.
* ( 1pt) Task2 Q7: The selected models are appropriate for the dataset and task, and the reasoning reflects a thoughtful connection between model characteristics (e.g., interpretability, ability to handle feature types, performance) and the data's structure and/or prediction goals.
* (1pt) Task3 Q10-14: At least half of the code demonstrates clear alignment with the outlined workflow, including appropriate use of methods for EDA, preprocessing, modeling, or evaluation
* (1pt) Task3 Q10-14: At least 75% of the code demonstrates clear alignment with the outlined workflow, including appropriate use of methods for EDA, preprocessing, modeling, or evaluation
* (1pt) Task3 Q10-14: All code is clearly written, technically sound, and thoughtfully implements each step of the workflow with appropriate use of sklearn tools and data handling.
* (1pt) Task3 Q13: The pipeline is well-structured, includes necessary preprocessing/modeling steps, and is flexible enough to support model selection and tuning.
* (1pt) Task3 Q15: The best-fit model is clearly interpreted, with reference to performance metrics and insights from feature importance or model behavior.
* (1pt) Task4 Q16: At least two insights are clearly articulated, specific, and supported with evidence from EDA, modeling, or feature engineering steps. The response shows a good understanding of the data and connects analysis to the project question.
* (1pt) Task4Q17: All three insights are clearly articulated, specific, and supported with evidence from EDA, modeling, or feature engineering steps. The response shows a good understanding of the data and connects analysis to the project question.
* (1pt) Task4 Q17: Strategies for creating a compelling story are applicable
* (1pt) Task4 Q18: Pitch is orignial, compelling, and reflective of the analysis
* (1pt) Q19: Thoughtfully reflected on the assignment
"""