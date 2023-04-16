from mimetypes import init
from posixpath import split
from pyexpat import features
from statistics import median
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import the data files into dataframes
def readData():
    dfTrain = pd.read_csv('Data/train.csv')
    dfTest = pd.read_csv('Data/test.csv')
    return dfTrain, dfTest

# initial exploration of data types, missing values, etc.
def initialAnalysis(df):
    print('Info:\n', df.info())
    print('Describe:\n', df.describe())
    nunique = df.nunique()
    print('Unique:\n', nunique)
    print('Count of null values per column:\n', df.isnull().sum().sort_values(ascending=False))
    return nunique

def columnNames(df):
    #print(list(df))
    return list(df)

# combine dataframes, if necessary
def combineData(df1, df2):
    combine = pd.concat([df1,df2], axis=0).reset_index(drop=True)
    return combine

def splitData(dfCom):
    dfTrain = dfCom[dfCom['Survived'].isin([1,0])]
    dfTest = dfCom[dfCom['Survived'].isnull()]
    return dfTrain, dfTest

# replace missing data in a column with mode of that column
def replaceWithMode(df, col):
    mode = df[col].dropna().mode()[0]
    df[col].fillna(mode, inplace=True)
    return df

# replace missing data in a column with mean of that column
def replaceWithMean(df, col):
    mean = df[col].dropna().mean()
    df[col].fillna(mean, inplace=True)
    return df

# replace missing data in a column with median of that column
def replaceWithMedian(df, col):
    median = df[col].dropna().median()
    df[col].fillna(median, inplace=True)
    return df

def replaceAgeNaN(df):
    age_nan_indices = list(df[df['Age'].isnull()].index)
    for index in age_nan_indices:
        median_age = df['Age'].median()
        predict_age = df['Age'][(df['SibSp'] == df.iloc[index]['SibSp']) & (df['Parch'] == df.iloc[index]['Parch'])& (df['Pclass'] == df.iloc[index]["Pclass"])].median()
        if np.isnan(predict_age):
            df.loc[df['Age'].isnull(), 'Age'] = median_age
        else:
            df.loc[df['Age'].isnull(), 'Age'] = predict_age
    #print(df['Age'])
    return df

# change second option to histogram. one for survivors and one for people who died 
def exploratoryAnalysis(df, nunique):
    columnNames = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    for col in columnNames:
        if nunique[col] < 10:
            print(df[col].value_counts(dropna=False).sort_values(ascending=False))
            plt.figure(num = None, figsize = (12, 16), dpi = 80, facecolor = 'w', edgecolor = 'k')
            plt.subplot(1, 2, 1)
            sns.barplot(x=col, y='Survived', data=df)
            plt.ylabel('Survival Probability')
            plt.title(f'Survival Probability by {col}')
            plt.subplot(1, 2, 2)
            df[col].value_counts(dropna=False).sort_values(ascending=False).plot.bar()
            plt.ylabel('Count')
            plt.title(f'Category Count by {col}')
            plt.show()

        else:
            df['tempCol'] = pd.cut(df[col], 10).value_counts(dropna=False).sort_values(ascending=False)
            print(pd.cut(df[col], 10).value_counts(dropna=False).sort_values(ascending=False))
            plt.figure(num = None, figsize=(12,16), dpi = 80, facecolor ='w', edgecolor = 'k')
            plt.subplot(1,2,1)
            sns.barplot(x='tempCol', y='Survived', data=df)
            plt.ylabel('Survival Probability')
            plt.title(f'Survival Probability by {col}')
            plt.subplot(1,2,2)
            pd.cut(df[col], 10).value_counts(dropna=False).sort_values(ascending=False).plot.bar()
            plt.ylabel('Count')
            plt.title(f'Category Count by {col}')
            df = df.drop('tempCol', axis=1)
    return columnNames

def isAlone(df):
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 0
    df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
    return df

def getDummies(df, col):
    df = pd.get_dummies(df, columns=[col])
    #print(df)
    return df

def makeAgeRanges(df, col):
    df['AgeBand'] = pd.cut(df[col], 10)
    #print(df['AgeBand'].value_counts())
    df.loc[df[col] <= 16.136, col] = 0
    df.loc[(df[col] > 16.136) & (df[col] <= 32.102), col] = 1
    df.loc[(df[col] > 32.102) & (df[col] <= 48.068), col] = 2
    df.loc[(df[col] > 48.068) & (df[col] <= 64.034), col] = 3
    df.loc[(df[col] > 64.034) & (df[col] <= 80.0), col] = 4
    return df

def makeFareRanges(df, col):
    df['Fare'] = df['Fare'].map(lambda x: np.log(x) if x > 0 else 0)
    df['FareBand'] = pd.cut(df[col], 10)
    #print(df['FareBand'].value_counts())
    df.loc[df[col] <= 1.248, col] = 0
    df.loc[(df[col] > 1.248) & (df[col] <= 2.496), col] = 1
    df.loc[(df[col] > 2.496) & (df[col] <= 3.743), col] = 2
    df.loc[(df[col] > 3.743) & (df[col] <= 4.991), col] = 3
    df.loc[df[col] > 4.991, col] = 4
    return df

def titleWork(df):
    df['Title'] = [name.split(',')[1].split('.')[0].strip() for name in df['Name']]
    df['Title'] = df['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Lady', 'Jonkheer', 'Don', 'Capt', 'the Countess', 'Sir', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    return df


# ADD A FUNC FOR A CORRELATION MATRIX
def modelFunc(dfTr, dfTe, features):
    y = dfTr.Survived
    #X = pd.get_dummies(dfTr[features])
    X = dfTr[features]
    X.to_csv('XTrain.csv', index=False)
    
    #X_test = pd.get_dummies(dfTe[features])
    X_test = dfTe[features]

    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
    model.fit(X,y)
    
    predictions = model.predict(X_test)
    
    output = pd.DataFrame({'PassengerId': dfTe.PassengerId,'Survived': predictions})
    output = output.astype({'Survived': 'int64'})
    print(output.dtypes)
    
    output.to_csv('TitanicOutput.csv', index=False)
    print('Submission saved!')


def xgModel(dfTr, dfTe, features):
    y = dfTr.Survived
    X = dfTr[features]
    X_test = dfTe[features]

    #params = {"max_depth": 9, "learning_rate": 0.02, "gamma": 0.1, "reg_lambda": 1, "scale_pos_weight": 1, "subsample": 0.5, "colsample_bytree": 0.7, "objective": 'binary:logistic'}
    params = {"max_depth": 5, "learning_rate": 0.01, "gamma": 0.4, "reg_lambda": 0.8, "scale_pos_weight": 2, "subsample": 0.9, "colsample_bytree": 0.9, "objective": 'binary:logistic'}

    model = xgb.XGBClassifier(**params)
    model.fit(X, y)

    preds = model.predict(X_test)
    output = pd.DataFrame({'PassengerId': dfTe.PassengerId,'Survived': preds})
    output = output.astype({'Survived': 'int64'})
    #print(output.dtypes)
    
    output.to_csv('TitanicOutputXG.csv', index=False)
    print('Submission saved!')

def xgModelSplit(df, features):
    X_train, X_test, y_train, y_test = train_test_split(df[features], df.Survived, stratify=df.Survived, random_state=1121218)
    params = {"max_depth": 5, "learning_rate": 0.01, "gamma": 0.4, "reg_lambda": 0.8, "scale_pos_weight": 2, "subsample": 0.9, "colsample_bytree": 0.9, "objective": 'binary:logistic'}

    model = xgb.XGBClassifier(**params)

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print(accuracy_score(y_test, preds))

def xgGridSearch(df, features):
    X_train, X_test, y_train, y_test = train_test_split(df[features], df.Survived, stratify=df.Survived, random_state=1121218)
    #param_grid = {"max_depth": [1, 5, 10], "learning_rate": [0.01, 0.1, 0.2], "gamma": [0.1, 0.25, 0.5], "reg_lambda": [0, 0.5, 1], "scale_pos_weight": [1, 2, 3], "subsample": [0.5, 0.7, 0.9], "colsample_bytree": [0.5, 0.7, 0.9]}
    param_grid = {"max_depth": [4, 5, 6], "learning_rate": [0.01, 0.02, 0.03], "gamma": [0.3, 0.4, 0.5], "reg_lambda": [0.8, 0.9, 1], "scale_pos_weight": [1.5, 2, 2.5], "subsample": [0.7, 0.8, 0.9], "colsample_bytree": [0.7, 0.8, 0.9]}
    model = xgb.XGBClassifier(objective="binary:logistic")
    gridCV = GridSearchCV(model, param_grid, n_jobs=-1, cv=3, scoring="roc_auc")

    gridCV.fit(df[features], df.Survived)
    print(gridCV.best_params_)
    print(gridCV.score(X_train, y_train))

def outlierFunc(df, column):
    upperLimit = df[column].mean() + 3*df[column].std()
    #print(upperLimit)
    lowerLimit = df[column].mean() - 3*df[column].std()
    if lowerLimit < 0:
        lowerLimit = 0
    #print(lowerLimit)
    df.loc[df[column] > upperLimit, column] = upperLimit
    df.loc[df[column] < lowerLimit, column] = lowerLimit
    #print(df.describe())
    return df

def dropOutliers(df, col):
    upperLimit = df[col].mean() + 3*df[col].std()
    #print(upperLimit)
    lowerLimit = df[col].mean() - 3*df[col].std()
    if lowerLimit < 0:
        lowerLimit = 0
    df = df[df[col] > lowerLimit]
    df = df[df[col] < upperLimit]
    return df

def main():
    
    dfTrain, dfTest = readData()
    #dfTrain = dropOutliers(dfTrain, 'Age')
    #dfTrain = dropOutliers(dfTrain, 'Fare')
    combined = combineData(dfTrain, dfTest)
    notUnique = initialAnalysis(combined)
    combined = replaceWithMean(combined, 'Fare')
    combined = replaceWithMode(combined, 'Embarked')
    combined = replaceAgeNaN(combined)
    combined = isAlone(combined)
    combined = getDummies(combined, 'Sex')
    combined = getDummies(combined, 'Embarked')
    combined = titleWork(combined)
    combined = getDummies(combined, 'Title')
    features = columnNames(combined)
    #combined = outlierFunc(combined, 'Age')
    #combined = outlierFunc(combined, 'Fare')
    #combined['Sex'] = combined['Sex'].map({'male': 0, 'female': 1})
    #features = ['Pclass', 'Age', 'Fare', 'FamilySize', 'Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare']
    features = ['Pclass', 'Age', 'Fare', 'FamilySize', 'Sex_male', 'Sex_female', 'Embarked_C', 'Embarked_Q', 'Embarked_S', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs', 'Title_Rare']
    dfTrain, dfTest = splitData(combined)
    #modelFunc(dfTrain,dfTest, features) 
    #xgModel(dfTrain, dfTest, features)  
    xgModelSplit(dfTrain, features) 
    #xgGridSearch(dfTrain, features)
#try not using dummies for title but instead replacing the title with the average survival rate of each value
#try not using get dummies for embarked or title but replace to numbers. also try replacing embarked with average survival rate
#try remapping the sex column again and getting rid of the two columns
#do a correlation matrix for final columns
main()