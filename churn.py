import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

#pd.set_option('display.max_columns', None)
#pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv("Churn_Modelling.csv")
########################################################################################################################
# CreditScore: Müşterinin kredi notu. 0 ile 1000 arasında bir değer alır. Yüksek kredi notu, müşterinin kredi
#              geçmişinin iyi olduğunu gösterir.
# Geography: Müşterinin yaşadığı ülke. Fransa, İspanya veya Almanya olabilir.
# Gender: Müşterinin cinsiyeti. Erkek veya Kadın olabilir.
# Age: Müşterinin yaşı. Sayısal bir değer alır.
# Tenure: Müşterinin bankayla olan ilişkisinin süresi. Aylık olarak ifade edilir. Sayısal bir değer alır.
# Balance: Müşterinin banka hesabındaki bakiye. Para birimi olarak Euro kullanılır. Sayısal bir değer alır.
# NumOfProducts: Müşterinin bankadan aldığı ürün sayısı. Kredi kartı, kredi, sigorta vb. olabilir.
# HasCrCard: Müşterinin bankadan kredi kartı olup olmadığını gösterir. 1 ise var, 0 ise yok anlamına gelir.
# IsActiveMember: Müşterinin bankayla aktif olarak iletişim halinde olup olmadığını gösterir.
#                 1 ise aktif, 0 ise pasif anlamına gelir.
# EstimatedSalary: Müşterinin tahmini yıllık geliri. Para birimi olarak Euro kullanılır. Sayısal bir değer alır.
# Exited: Müşterinin bankayı terk edip etmediğini gösterir. 1 ise terk etti, 0 ise terk etmedi anlamına gelir.
########################################################################################################################
print(df.columns)
df = df.drop(["CustomerId", "RowNumber", "Surname"], axis=1)
print(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df)
#print(cat_cols) #['Geography', 'Gender', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited']
#print(num_cols) #['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
#print(cat_but_car)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


#for col in cat_cols:
#    cat_summary(df, col, True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


#for col in num_cols:
#    num_summary(df, col, True)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


#for col in num_cols:
#   target_summary_with_num(df, "Exited", col)


##################################
# AYKIRI DEĞER ANALİZİ
##################################
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


#for col in num_cols:
#    print(f"{col} : {check_outlier(df, col)}")


##################################
# ENCODING
##################################
df = pd.get_dummies(df, drop_first=True)
print(df)


##################################
# BASE MODEL
##################################
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
y = df["Exited"]
X = df.drop("Exited", axis=1)

log_model = LogisticRegression().fit(X, y)
y_pred = log_model.predict(X)
print(pd.DataFrame(y_pred).value_counts())
print(df["Exited"].value_counts())

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("Accuracy Score: {0}".format(acc), size=10)
    plt.show()


#plot_confusion_matrix(y, y_pred)
print(classification_report(y, y_pred))
y_prob = log_model.predict_proba(X)[:, 1]
print(roc_auc_score(y, y_prob))
# accuracy = 0.79
# precision = 0.39
# recall = 0.06
# f1-score = 0.10
# roc_auc_score = 0.670547341838921


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
log_model2 = LogisticRegression().fit(X_train, y_train)
y_pred2 = log_model.predict(X_test)
y_prob2 = log_model2.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred2))
print(roc_auc_score(y_test, y_prob2))
# accuracy = 0.78
# precision = 0.22
# recall = 0.02
# f1-score = 0.04
# roc_auc_score = 0.6550325614830248


cv_result = cross_validate(log_model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])
print(cv_result["test_accuracy"].mean())
# accuracy = 0.7907
# precision = 0.40471473408356395
# recall = 0.0589066339066339
# f1-score = 0.10283523854039023
# roc_auc_score = 0.6723354789673579