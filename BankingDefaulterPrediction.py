
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix


############  Data Loading Part #############
#Loading the Query result data from MS Access output 
# the query generates Loan Status and Distric Data together along with Account Id 

df = pd.read_excel('LoanToDemographDistinctMap.xls')
df.head() 

#################### Data Cleansing Part ######################

# NaN value treatment 
df.fillna(value=0, inplace=True)

#Remove the non-numeric data to prepare it for KNN Analaysis
DemoGraphNumeric = df.drop(['account_id', 'ID', 'A1','A2','A3','status'] , axis=1)

# Need to make all the numeric data in a same scale 
scaler = StandardScaler()
scaler.fit(DemoGraphNumeric)
scaled_DemoGraphNumeric = scaler.transform(DemoGraphNumeric)
scaled_DemoGraphNumeric_feature = pd.DataFrame(scaled_DemoGraphNumeric,columns=DemoGraphNumeric.columns)


#Split the training and testing data

X_train, X_test, y_train, y_test = train_test_split(scaled_DemoGraphNumeric_feature,df['status'],
                                                    test_size=0.99)
# Initialize KNN with K=2


knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(scaled_DemoGraphNumeric_feature,df['status'])


Status_prediction = knn.predict(X_test)

set(y_test) - set(Status_prediction )

print(confusion_matrix(y_test,Status_prediction))



print(classification_report(y_test,Status_prediction))


# # Choosing a K Value

error_rate = []

# Will take some time
for i in range(1,60):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(scaled_DemoGraphNumeric_feature,df['status'])
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


plt.figure(figsize=(10,6))
plt.plot(range(1,60),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# ## Retrain with new K Value
# 
# **Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.**



# NOW WITH K=30
#knn = KNeighborsClassifier(n_neighbors=30)

#knn.fit(X_train,y_train)
#pred = knn.predict(X_test)

#print('WITH K=30')
#print('\n')
#print(confusion_matrix(y_test,pred))
#print('\n')
#print(classification_report(y_test,pred))


# # Great Job!
