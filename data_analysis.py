import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle


def is_real_and_finite(x):
    try: 
        float (x)
        return True
    except:
        return False
   

def getData (data_url):
    df = pd.read_csv(data_url)
    
    categories = df['category'].unique()
    merchants = df['merchant'].unique()
    mf = df['gender'].unique()
    #age = df['age'].unique()
    customer = df['customer'].unique()
    
    cat_keys = makeDict(categories) #making a dictionary with 0 to n as keys, each corresponding to unique value in for each feature
    merch_keys= makeDict (merchants)
    mf_keys = makeDict(mf)
    cust_keys = makeDict(customer)
    
    #print (cat_keys)
    
    
 
    replaceValues(cat_keys, df,'category' )
    replaceValues(merch_keys, df,'merchant' )
    replaceValues(mf_keys, df,'gender' )
    #replaceValues(cust_keys, df,'customer' ) #this lines takes really long because 4112 unique customer names, so just truncate at C 
    df['customer'] = df ['customer'].str.replace("'", '', regex = False)
    df['customer'] = df ['customer'].str.replace("C", '', regex = False) #strips leading C in customer column
    df['age'] = df ['age'].str.replace("'", '', regex = False)
    df['zipcodeOri'] = df ['zipcodeOri'].str.replace("'", '', regex = False)
    df['zipMerchant'] = df ['zipMerchant'].str.replace("'", '', regex = False)
   
    nonstepCols = df.columns [1:] #takes out first column which is BankSimulator 'step'
    numeric_map = df[nonstepCols].applymap(is_real_and_finite)
    numeric_rows = numeric_map.all(axis=1)

    real_rows = numeric_map.all(axis=1).values
    df_dropped_obs = df[real_rows]
   
    float_np = df_dropped_obs.values[:, 1:].astype('float')#np formatted in float without the first 'step' column
    
    float_df = pd.DataFrame(float_np, columns = [df.columns[1:10]])
    print (float_df)
    return float_df

def replaceValues (keyValues, df, colName):
    """[summary]

    Args:
        keyValue ([type]): [description]
        df ([type]): [description]
        colName [str]: column name to do replacements in the unedited pandas df
    """
    for key in keyValues:
       
        df[colName] = df[colName].replace([keyValues[key]],key )
 
def makeDict (array):
    #assigns unique values to dictionary value 
    d = {}
    for i in range(len(array)):
        d[i] = array[i]
    #print (len(d))
    return d

def acc_prec_recall(y_model, y_actual):
    TP = np.sum(np.logical_and(y_model == y_actual, y_model == 1))
    TN = np.sum(np.logical_and(y_model == y_actual, y_model == 0))
    FP = np.sum(np.logical_and(y_model != y_actual, y_model == 1))
    FN = np.sum(np.logical_and(y_model != y_actual, y_model == 0))
    acc = (TP + TN) / (TP + TN + FP + FN)
    if TP == 0:
        prec = 0
        recall = 0
    else:
        prec = TP / (TP + FP)
        recall = TP / (TP + FN)
    return acc, prec, recall

def kNN (df):
    #df = df.astype(float)
 
    X = df.iloc[:,:8]
  
    y = df.iloc [:, 8:9]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train.values.ravel())
    y_knn = knn.predict(X_test)#return np array
    
   
    acc, prec, recall = acc_prec_recall(y_knn.reshape(len(y_knn),1), y_test.values.reshape(y_test.size,1))

    print ("Accuracy: ", acc, " Precision: ", prec, " Recall: ", recall)
    print(knn.score(X_test, y_test))
    
def SVM (df):
    X = df.iloc[:,:8]
  
    y = df.iloc [:, 8:9]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    sigmas = np.array([0.001, 0.1, 1])
    gammas = 1./(2*sigmas**2)

    alphas = np.array([1e-9, 1e-5, 1e-4])
    Cs = 1/alphas
     
    parameter_ranges = {'C':Cs, 'gamma':gammas}

    svc = SVC(kernel='rbf') #has 386 weights (from 385 features and 1 intercept)

    svc_search = GridSearchCV(svc, parameter_ranges, cv=3)
    svc_search.fit(X_train,y_train.values.ravel())
    print(svc_search.best_estimator_, svc_search.best_score_) #default score it prints is accuracy

    best_svc = svc_search.best_estimator_

    y_predict = best_svc.predict(X_test)
    acc, prec, recall = acc_prec_recall(y_predict, y_test)
    print ("Accuracy: ", acc, " Precision: ", prec, " Recall: ", recall)

    print(best_svc.score(X_test, y_test))

def is_category (float_df, numCategories):
    float_np = float_df.values
    arrays = {}
    
    for catNum in range (numCategories):
        arrays[catNum] = []
        for i in range(len(float_np)):
            if float_np[i][0] == float(catNum):
                arrays[catNum].append (float_np[i][1])
              
    
    return arrays

    
def getColHist(df, col = 7): #for category):
    df = pd.read_csv(data_url)
    
    categories = df['category'].unique()
    
    cat_keys = makeDict(categories) #making a dictionary with 0 to n as keys, each corresponding to unique value in for each feature
    
    print(cat_keys)
 
    replaceValues(cat_keys, df,'category' )
    
    #replaceValues(cust_keys, df,'customer' ) #this lines takes really long because 4112 unique customer names, so just truncate at C 
  
    nonstepCols = df.columns [col:9] #keep only category and amount
    numeric_map = df[nonstepCols].applymap(is_real_and_finite)
    numeric_rows = numeric_map.all(axis=1)
    real_rows = numeric_map.all(axis=1).values
    df_dropped_obs = df[real_rows]

   
    float_np = df_dropped_obs.values[:, col:9].astype('float')#np formatted in float without the first 'step' column
    
    float_df = pd.DataFrame(float_np, columns = ['category', 'amount'])

    return float_df

if __name__ == "__main__":
    data_url = 'https://raw.githubusercontent.com/ayw7af/hackathon/main/bs140513_032310.csv'
    #df = getData (data_url)
    df = getColHist(data_url)
    d = is_category (df, 15)
    print(d)
    #kNN (df)
    #SVM (df)