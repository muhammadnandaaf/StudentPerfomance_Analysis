#!/usr/bin/env python
# coding: utf-8

# # **Submission Akhir : Menyelesaikan Permasalahan Institusi Pendidikan**
# - Muhammad Nandaarjuna Fadhillah
# - muhammadnandaaf@gmail.com
# - muhammadnanda

# ## **Import Library**

# In[43]:


import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import numpy as np 
import pandas as pd 
import math

from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
import collections
from collections import Counter

from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle

from sklearn.preprocessing import LabelEncoder,QuantileTransformer,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score, classification_report

import joblib


# ## **Data Loading**

# In[44]:


url = "https://raw.githubusercontent.com/dicodingacademy/dicoding_dataset/refs/heads/main/students_performance/data.csv"
dataset = pd.read_csv(url, delimiter=';')
dataset.head()


# ## **Data Understanding**

# In[45]:


dataset.shape


# ### **Exploratory Data Analysis**

# In[46]:


explore_df = dataset.copy()
explore_df.head().T


# ### **Data Statistics**

# In[47]:


print("Statistics Numerics Results")
display(dataset.describe().T)

print("Statistics of all columns")
display(dataset.describe(include='all').T)


# ### **Data Condition**

# In[48]:


print("\nDataset Row Duplicated:")
display(explore_df.duplicated().sum())

print("\nDataset Information:")
explore_df.info()

print("\nDuplicated Data per Column:")
for col in explore_df.columns:
    print(f"{col:20} : {explore_df[col].duplicated().sum()}")

print("\nMissing Values per Column:")
for col in explore_df.columns:
    print(f"{col:20} : {explore_df[col].isna().sum()}")


# ### **Prepare Data Exploration**

# In[49]:


# Mapping
marital_status_map = {
    1: "Single", 2: "Married", 3: "Widower", 4: "Divorced", 5: "Facto Union", 6: "Legally Separated"
}
gender_map = {1: "Male", 0: "Female"}
daytime_evening_map = {1: "Daytime", 0: "Evening"}
displaced_map = {1: "Yes", 0: "No"}
Educational_special_needs_map = {1: "Yes", 0: "No"}
Tuition_fees_up_to_date_map = {1: "Yes", 0: "No"}
Scholarship_holder_map = {1: "Yes", 0: "No"}
International_map = {1: "Yes", 0: "No"}
binary_map = {1: "Yes", 0: "No"}

application_mode_map = {
    1: "1st phase - general contingent", 2: "Ordinance No. 612/93", 5: "1st phase - special contingent (Azores Island)",
    7: "Holders of other higher courses", 10: "Ordinance No. 854-B/99", 15: "International student (bachelor)",
    16: "1st phase - special contingent (Madeira Island)", 17: "2nd phase - general contingent",
    18: "3rd phase - general contingent", 26: "Ordinance No. 533-A/99, item b2) (Different Plan)",
    27: "Ordinance No. 533-A/99, item b3 (Other Institution)", 39: "Over 23 years old", 42: "Transfer",
    43: "Change of course", 44: "Technological specialization diploma holders", 51: "Change of institution/course",
    53: "Short cycle diploma holders", 57: "Change of institution/course (International)"
}
course_map = {
    33: "Biofuel Production Technologies", 171: "Animation and Multimedia Design", 8014: "Social Service (evening attendance)",
    9003: "Agronomy", 9070: "Communication Design", 9085: "Veterinary Nursing", 9119: "Informatics Engineering",
    9130: "Equinculture", 9147: "Management", 9238: "Social Service", 9254: "Tourism", 9500: "Nursing",
    9556: "Oral Hygiene", 9670: "Advertising and Marketing Management", 9773: "Journalism and Communication",
    9853: "Basic Education", 9991: "Management (evening attendance)"
}
previous_qualification_map = {
    1: "Secondary education", 2: "Higher education - bachelor's degree", 3: "Higher education - degree",
    4: "Higher education - master's", 5: "Higher education - doctorate", 6: "Frequency of higher education",
    9: "12th year - not completed", 10: "11th year - not completed", 12: "Other - 11th year", 14: "10th year",
    15: "10th year - not completed", 19: "Basic education 3rd cycle", 38: "Basic education 2nd cycle",
    39: "Technological specialization course", 40: "Higher education - degree (1st cycle)",
    42: "Professional higher technical course", 43: "Higher education - master (2nd cycle)"
}
nacionality_map = {
    1: "Portuguese", 2: "German", 6: "Spanish", 11: "Italian", 13: "Dutch", 14: "English", 17: "Lithuanian",
    21: "Angolan", 22: "Cape Verdean", 24: "Guinean", 25: "Mozambican", 26: "Santomean", 32: "Turkish",
    41: "Brazilian", 62: "Romanian", 100: "Moldovan", 101: "Mexican", 103: "Ukrainian", 105: "Russian",
    108: "Cuban", 109: "Colombian"
}
mothers_qualification_map = {
    1: "Secondary Education - 12th Year", 2: "Higher Education - Bachelor's", 3: "Higher Education - Degree",
    4: "Higher Education - Master's", 5: "Higher Education - Doctorate", 6: "Frequency of Higher Education",
    9: "12th Year - Not Completed", 10: "11th Year - Not Completed", 11: "7th Year (Old)", 12: "Other - 11th Year",
    14: "10th Year", 15: "10th Year - Not Completed", 18: "General Commerce Course",
    19: "Basic Education 3rd Cycle", 22: "Technical-Professional Course", 26: "7th Year of Schooling",
    27: "2nd Cycle of General High School", 29: "9th Year - Not Completed", 30: "8th Year",
    34: "Unknown", 35: "Can't Read or Write", 36: "Can Read (No 4th Year)", 37: "Basic Education 1st Cycle",
    38: "Basic Education 2nd Cycle", 39: "Technological Specialization Course", 40: "Degree (1st Cycle)",
    41: "Specialized Higher Studies", 42: "Professional Higher Technical Course", 43: "Master (2nd Cycle)",
    44: "Doctorate (3rd Cycle)"
}
fathers_qualification_map = {
    1: "Secondary Education - 12th Year", 2: "Higher Education - Bachelor's", 3: "Higher Education - Degree",
    4: "Higher Education - Master's", 5: "Higher Education - Doctorate", 6: "Frequency of Higher Education",
    9: "12th Year - Not Completed", 10: "11th Year - Not Completed", 11: "7th Year (Old)", 12: "Other - 11th Year",
    13: "2nd Year Complementary HS", 14: "10th Year", 18: "General Commerce Course", 19: "Basic Education 3rd Cycle",
    20: "Complementary HS", 22: "Technical-Professional Course", 25: "Complementary HS - Not Completed",
    26: "7th Year", 27: "2nd Cycle General HS", 29: "9th Year - Not Completed", 30: "8th Year",
    31: "Admin and Commerce General", 33: "Supplementary Accounting/Admin", 34: "Unknown",
    35: "Can't Read or Write", 36: "Can Read (No 4th Year)", 37: "Basic Education 1st Cycle",
    38: "Basic Education 2nd Cycle", 39: "Technological Specialization Course", 40: "Degree (1st Cycle)",
    41: "Specialized Higher Studies", 42: "Professional Higher Technical Course", 43: "Master (2nd Cycle)",
    44: "Doctorate (3rd Cycle)"
}
mothers_occupation_map = {
    0: "Student", 1: "Executive/Director", 2: "Intellectual/Scientific", 3: "Technician/Professional",
    4: "Administrative", 5: "Service/Security/Sales", 6: "Agriculture/Fishery", 7: "Construction/Craft",
    8: "Machine Operator", 9: "Unskilled", 10: "Armed Forces", 90: "Other", 99: "Blank", 122: "Health Professionals",
    123: "Teacher", 125: "ICT Specialist", 131: "Science & Engineering Technician", 132: "Health Technician",
    134: "Legal/Social/Sports Technician", 141: "Secretary/Data Operator", 143: "Finance/Registry Operator",
    144: "Other Admin", 151: "Service Worker", 152: "Seller", 153: "Personal Care", 171: "Construction Worker",
    173: "Printing/Precision/Craft", 175: "Food/Wood/Clothing", 191: "Cleaner",
    192: "Unskilled Agriculture", 193: "Unskilled Industry", 194: "Meal Assistant"
}
fathers_occupation_map = {
    0: "Student", 1: "Executive/Director", 2: "Intellectual/Scientific", 3: "Technician/Professional",
    4: "Administrative", 5: "Service/Security/Sales", 6: "Agriculture/Fishery", 7: "Construction/Craft",
    8: "Machine Operator", 9: "Unskilled", 10: "Armed Forces", 90: "Other", 99: "Blank", 101: "Armed Forces Officer",
    102: "Sergeant", 103: "Other Armed Forces", 112: "Admin Director", 114: "Hotel/Trade Director",
    121: "Physics/Engineering Specialist", 122: "Health Professional", 123: "Teacher",
    124: "Finance/Admin/Public Relations", 131: "Science & Engineering Technician", 132: "Health Technician",
    134: "Legal/Social/Sports Technician", 135: "ICT Technician", 141: "Secretary/Data Operator",
    143: "Finance/Registry Operator", 144: "Other Admin", 151: "Service Worker", 152: "Seller",
    153: "Personal Care", 154: "Security", 161: "Skilled Farmer", 163: "Subsistence Farmer",
    171: "Construction Worker", 172: "Metalworker", 174: "Electrician/Electronics", 175: "Food/Wood/Clothing",
    181: "Fixed Machine Operator", 182: "Assembler", 183: "Driver", 192: "Unskilled Agriculture",
    193: "Unskilled Industry", 194: "Meal Assistant", 195: "Street Vendor"
}


# In[50]:


explore_df["Marital_status"] = explore_df["Marital_status"].map(marital_status_map)
explore_df["Application_mode"] = explore_df["Application_mode"].map(application_mode_map)
explore_df["Course"] = explore_df["Course"].map(course_map)
explore_df["Daytime_evening_attendance"] = explore_df["Daytime_evening_attendance"].map(daytime_evening_map)
explore_df["Previous_qualification"] = explore_df["Previous_qualification"].map(previous_qualification_map)
explore_df["Nacionality"] = explore_df["Nacionality"].map(nacionality_map)
explore_df["Displaced"] = explore_df["Displaced"].map(binary_map)
explore_df["Educational_special_needs"] = explore_df["Educational_special_needs"].map(binary_map)
explore_df["Debtor"] = explore_df["Debtor"].map(binary_map)
explore_df["Tuition_fees_up_to_date"] = explore_df["Tuition_fees_up_to_date"].map(binary_map)
explore_df["Gender"] = explore_df["Gender"].map(gender_map)
explore_df["Scholarship_holder"] = explore_df["Scholarship_holder"].map(binary_map)
explore_df["International"] = explore_df["International"].map(binary_map)
explore_df["Mothers_qualification"] = explore_df["Mothers_qualification"].map(mothers_qualification_map)
explore_df["Fathers_qualification"] = explore_df["Fathers_qualification"].map(fathers_qualification_map)
explore_df["Mothers_occupation"] = explore_df["Mothers_occupation"].map(mothers_occupation_map)
explore_df["Fathers_occupation"] = explore_df["Fathers_occupation"].map(fathers_occupation_map)
explore_df.head().T


# In[51]:


numerical_columns = explore_df.select_dtypes(include=["int64", "float64"]).columns

# Ubah ke dalam dataframe
numerical_df = pd.DataFrame(explore_df[numerical_columns])


# ### **Outlier Check**

# In[52]:


outlier_summary = {}
# Set style
sns.set(style="whitegrid")

# Fungsi deteksi outlier dengan IQR
def detect_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# Jumlah kolom dalam grid
n_cols = 3
# Gunakan pembulatan ke atas agar cukup tempat untuk semua variabel
n_rows = math.ceil(len(numerical_columns) / n_cols)
n_cols = 3
n_rows = math.ceil(len(numerical_columns) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

# Loop untuk membuat boxplot tiap variabel
for i, col in enumerate(numerical_columns):
    sns.boxplot(data=dataset, x=col, ax=axes[i], color='skyblue')
    outliers = detect_outliers_iqr(dataset, col)
    if len(outliers) > 0:
        outlier_summary[col] = len(outliers)
    axes[i].set_title(f"{col} ({len(outliers)} outliers)", fontsize=11)
    axes[i].set_xlabel("")
    axes[i].grid(True)

# Hapus subplot kosong jika ada
for j in range(len(numerical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

print("Outliers Column:")
for col, count in outlier_summary.items():
    print(f"{col}: {count} outliers")


# In[53]:


for col in outlier_summary.keys():
    outliers = detect_outliers_iqr(dataset, col)
    print(f"\nOutliers for '{col}':")
    print(outliers[[col]].drop_duplicates().sort_values(by=col))


# ### **Visualization & Correlation Analysis**

# **Numerical Column Distribution**

# In[54]:


plt.figure(figsize=(20, 20))
for idx, col in enumerate(numerical_df.columns):
    plt.subplot(6, 5, idx+1)
    sns.histplot(data=explore_df, x=col, kde=False)
    plt.title(f'Distribusi {col}')
plt.tight_layout()
plt.show()


# **Categorical Column Distribution**

# In[55]:


categorical_columns = explore_df.select_dtypes(include=["object"]).columns
categorical_df = pd.DataFrame(explore_df[categorical_columns])

binary_columns = [
    "Gender", "Daytime_evening_attendance", "Displaced", "Educational_special_needs",
    "Debtor", "Tuition_fees_up_to_date", "Scholarship_holder", "International"
]


# In[56]:


n_cols = 3
n_rows = math.ceil(len(binary_columns) / n_cols)

plt.figure(figsize=(5 * n_cols, 5 * n_rows))

for idx, col in enumerate(binary_columns):
    plt.subplot(n_rows, n_cols, idx + 1)
    sns.countplot(data=explore_df, x=col, hue="Status", order=explore_df[col].value_counts().index, palette="Set2")
    plt.title(f"Distribusi {col}")
    plt.xlabel(col)
    plt.ylabel("Jumlah")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# In[57]:


def reduce_categories(df, column, top_n=10):
    top_categories = df[column].value_counts().nlargest(top_n).index
    return df[column].apply(lambda x: x if x in top_categories else 'Other')

multi_value_cats = [col for col in explore_df.select_dtypes(include='object').columns 
                    if explore_df[col].nunique() > 2 and col != 'Status']


# In[58]:


# Setup grid layout
n_cols = 2
n_rows = math.ceil(len(multi_value_cats) / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_cols, 5 * n_rows))
axes = axes.flatten()

for idx, col in enumerate(multi_value_cats):
    df_temp = explore_df.copy()
    df_temp[col] = reduce_categories(df_temp, col, top_n=10)

    # Crosstab untuk proporsi per status
    data_crosstab = pd.crosstab(df_temp[col], df_temp['Status'], normalize='index') * 100
    data_crosstab = data_crosstab[sorted(data_crosstab.columns)]  # Urutkan status
    data_crosstab.sort_index(inplace=True)

    # Deteksi apakah label terlalu panjang → bar horizontal
    is_horizontal = any(len(str(x)) > 20 for x in data_crosstab.index)

    ax = axes[idx]
    data_crosstab.plot(kind='barh' if is_horizontal else 'bar',
                       stacked=True,
                       ax=ax,
                       colormap='tab10',
                       width=0.85)

    ax.set_title(f'Stacked Bar: {col}', fontsize=12)
    ax.set_ylabel('Persentase (%)')
    ax.set_xlabel('Persentase (%)' if is_horizontal else '')
    ax.legend(title='Status', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='x' if is_horizontal else 'y', linestyle='--', alpha=0.5)

    if not is_horizontal:
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.tick_params(axis='y', labelsize=8)

# Hapus subplot kosong jika ada
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# **Heatmap Correlation**

# In[59]:


# Visualisasi matriks korelasi
plt.figure(figsize=(12, 8))
corr_matrix = explore_df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Heatmap Correlation Matrix")
plt.show()

corr_matrix


# ## **Data Preparation**

# In[60]:


model_df = explore_df.copy()
model_df.head().T


# ### **Splitting Data**

# In[61]:


train_df, test_df = train_test_split(model_df, test_size=0.05, random_state=42, shuffle=True)
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)
 
print(train_df.shape)
print(test_df.shape)


# In[62]:


sns.countplot(data=train_df, x="Status")
plt.show()
print(train_df.Status.value_counts())


# ### **Oversampling**

# In[63]:


df_majority_1 = train_df[(train_df.Status == "Graduate")]
df_majority_2 = train_df[(train_df.Status == "Dropout")]
df_minority = train_df[(train_df.Status == "Enrolled")]

df_majority_2_undersampled = resample(df_majority_2, n_samples=2101, random_state=42)
df_minority_undersampled = resample(df_minority, n_samples=2101, random_state=42)

print(df_majority_2_undersampled.shape)
print(df_minority_undersampled.shape)


# In[64]:


oversampled_train_df = pd.concat([df_majority_1, df_majority_2_undersampled]).reset_index(drop=True)
oversampled_train_df = pd.concat([oversampled_train_df, df_minority_undersampled]).reset_index(drop=True)
oversampled_train_df = shuffle(oversampled_train_df, random_state=42)
oversampled_train_df.reset_index(drop=True, inplace=True)
display(oversampled_train_df.sample(5))
sns.countplot(data=oversampled_train_df, x="Status")
plt.show()


# In[65]:


X_train = oversampled_train_df.drop(columns="Status", axis=1)
y_train = oversampled_train_df["Status"]
 
X_test = test_df.drop(columns="Status", axis=1)
y_test = test_df["Status"]


# ### **Scaling & Encoding**

# In[66]:


def scaling(features, df, df_test=None):
    os.makedirs("model", exist_ok=True)
    
    if df_test is not None:
        df = df.copy()
        df_test = df_test.copy()
        for feature in features:
            scaler = MinMaxScaler()
            X = np.asanyarray(df[feature])
            X = X.reshape(-1,1)
            scaler.fit(X)
            df["{}".format(feature)] = scaler.transform(X)
            joblib.dump(scaler, "model/scaler_{}.joblib".format(feature))
            
            X_test = np.asanyarray(df_test[feature])
            X_test = X_test.reshape(-1,1)
            df_test["{}".format(feature)] = scaler.transform(X_test)
        return df, df_test
    else:
        df = df.copy()
        for feature in features:
            scaler = MinMaxScaler()
            X = np.asanyarray(df[feature])
            X = X.reshape(-1,1)
            scaler.fit(X)
            df["{}".format(feature)] = scaler.transform(X)
            joblib.dump(scaler, "model/scaler_{}.joblib".format(feature))
        return df
 
def encoding(features, df, df_test=None):
    os.makedirs("model", exist_ok=True)

    if df_test is not None:
        df = df.copy()
        df_test = df_test.copy()
        for feature in features:
            encoder = LabelEncoder()
            
            # Gabungkan data training dan testing untuk menghindari unseen labels
            combined_data = pd.concat([df[feature], df_test[feature]], axis=0)
            encoder.fit(combined_data)

            df[feature] = encoder.transform(df[feature])
            df_test[feature] = encoder.transform(df_test[feature])
            
            joblib.dump(encoder, f"model/encoder_{feature}.joblib")

        return df, df_test
    else:
        df = df.copy()
        for feature in features:
            encoder = LabelEncoder()
            encoder.fit(df[feature])
            df[feature] = encoder.transform(df[feature])
            joblib.dump(encoder, f"model/encoder_{feature}.joblib")
        return df


# In[67]:


numerical_columns_model = X_train.select_dtypes(include=["int64", "float64"]).columns.drop('Status', errors='ignore')
categorical_columns_model = X_train.select_dtypes(include=["object"]).columns.drop('Status', errors='ignore')

# Ubah ke dalam dataframe
numerical_model_df = pd.DataFrame(model_df[numerical_columns_model])
categorical_model_df = pd.DataFrame(model_df[categorical_columns_model])


# In[68]:


new_train_df, new_test_df = scaling(numerical_columns_model, X_train, X_test)
new_train_df, new_test_df = encoding(categorical_columns_model, new_train_df, new_test_df)


# In[69]:


encoder = LabelEncoder()
encoder.fit(y_train)
new_y_train = encoder.transform(y_train)
joblib.dump(encoder, "model/encoder_target.joblib")
 
new_y_test = encoder.transform(y_test)


# ### **PCA (Principal Component Analysis)**

# In[70]:


pca_academic_semester_1 = [
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations'
]
pca_academic_semester_2 = [
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations'
]
train_pca_df = new_train_df.copy().reset_index(drop=True)
test_pca_df = new_test_df.copy().reset_index(drop=True)


# In[71]:


def apply_pca_and_transform(pca_features, train_df, test_df, pca_name):
    """ 
    Melakukan PCA dengan jumlah komponen efisien (min. 95% variansi), 
    menyimpan model, dan menambahkan principal components ke DataFrame.
    """
    # Inisialisasi PCA tanpa mengatur n_components
    pca = PCA(random_state=123)
    pca.fit(train_df[pca_features])

    # Hitung variansi kumulatif
    var_exp = pca.explained_variance_ratio_.round(3)
    cum_var_exp = np.cumsum(var_exp)
    
    # Tentukan jumlah komponen efisien (>=95% variansi kumulatif)
    n_components = np.argmax(cum_var_exp >= 0.95) + 1
    print(f"🔍 Optimal number of components for {pca_name}: {n_components} (cumulative variance ≥ 95%)")

    # Visualisasi Explained Variance
    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(var_exp) + 1), var_exp, alpha=0.5, align='center', label='Individual Variance')
    plt.step(range(1, len(cum_var_exp) + 1), cum_var_exp, where='mid', label='Cumulative Variance', color='red')
    plt.axhline(y=0.95, color='green', linestyle='--', label='95% Threshold')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Index')
    plt.title(f'PCA Explained Variance for {pca_name}')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Ulang PCA dengan n_components optimal
    pca = PCA(n_components=n_components, random_state=123)
    pca.fit(train_df[pca_features])
    
    # Simpan model PCA
    joblib.dump(pca, f"model/pca_{pca_name}.joblib")
    print(f"📌 PCA Model for '{pca_name}' saved as 'model/pca_{pca_name}.joblib'.")
    
    # Transform data
    princ_comp_train = pca.transform(train_df[pca_features])
    princ_comp_test = pca.transform(test_df[pca_features])

    # Membuat nama kolom baru
    pca_columns = [f"{pca_name}_{i+1}" for i in range(n_components)]
    
    # Tambahkan Principal Components ke DataFrame
    train_df[pca_columns] = pd.DataFrame(princ_comp_train, columns=pca_columns, index=train_df.index)
    test_df[pca_columns] = pd.DataFrame(princ_comp_test, columns=pca_columns, index=test_df.index)
    
    # Drop kolom asli yang digunakan di PCA
    train_df.drop(columns=pca_features, inplace=True)
    test_df.drop(columns=pca_features, inplace=True)
    print(f"\n📌 Features for '{pca_name}' successfully replaced with {n_components} Principal Components.\n")

    return train_df, test_df


# In[72]:


# Semester 1
train_pca_df, test_pca_df = apply_pca_and_transform(
    pca_features=pca_academic_semester_1, 
    train_df=train_pca_df, 
    test_df=test_pca_df,
    pca_name="pc1"
)

# Semester 2
train_pca_df, test_pca_df = apply_pca_and_transform(
    pca_features=pca_academic_semester_2, 
    train_df=train_pca_df, 
    test_df=test_pca_df,
    pca_name="pc2"
)


# In[73]:


# Tampilkan hasil akhir
print("\n📌 DataFrame setelah PCA:")
display(train_pca_df.head())
display(test_pca_df.head())


# ## **Model Development**

# ### **Hyperparameter Tuning**

# In[74]:


# Inisialisasi Stratified K-Fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ==================== DECISION TREE ====================
dt_params = {
    'max_depth': Integer(1, 50),          
    'min_samples_split': Integer(2, 20),   
    'criterion': Categorical(['gini', 'entropy']),  
}

dt_ori = DecisionTreeClassifier()
dt_bayes = BayesSearchCV(
    estimator=dt_ori,
    search_spaces=dt_params,
    scoring='accuracy', 
    cv=cv,
    n_iter=50,           
    n_jobs=-1,           
    random_state=42
)
dt_bayes.fit(train_pca_df, new_y_train)

# ==================== RANDOM FOREST ====================
rf_params = {
    'n_estimators': Integer(10, 1000),    
    'max_depth': Integer(1, 50),          
    'min_samples_split': Integer(2, 20),   
    'criterion': Categorical(['gini', 'entropy']),  
}

rf_ori = RandomForestClassifier()
rf_bayes = BayesSearchCV(
    estimator=rf_ori,
    search_spaces=rf_params,
    scoring='accuracy', 
    cv=cv,
    n_iter=50,           
    n_jobs=-1,           
    random_state=42
)
rf_bayes.fit(train_pca_df, new_y_train)

# ==================== NAIVE BAYES ====================
nb_params = {
    'var_smoothing': Real(1e-9, 1e-1, prior='log-uniform')
}

nb_ori = GaussianNB()
nb_bayes = BayesSearchCV(
    estimator=nb_ori,
    search_spaces=nb_params,
    scoring='accuracy',
    cv=cv,
    n_iter=20,  
    n_jobs=-1,
    random_state=42
)
nb_bayes.fit(train_pca_df, new_y_train)

# ==================== XGBOOST ====================
xgb_params = {
    'n_estimators': Integer(50, 1000),
    'max_depth': Integer(1, 50),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'subsample': Real(0.5, 1.0, prior='uniform'),
    'colsample_bytree': Real(0.5, 1.0, prior='uniform'),
    'gamma': Real(0, 5, prior='uniform')
}

xgb_ori = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_bayes = BayesSearchCV(
    estimator=xgb_ori,
    search_spaces=xgb_params,
    scoring='accuracy',
    cv=cv,
    n_iter=50,
    n_jobs=-1,
    random_state=42
)
xgb_bayes.fit(train_pca_df, new_y_train)

def print_tuning_results(models):
    """ 
    Fungsi untuk mencetak hasil tuning model dengan format yang lebih rapi.
    """
    print("\n📌 **Hasil Hyperparameter Tuning**")
    print("="*60)
    for name, model in models.items():
        print(f"\n🔹 {name} Best Parameters:")
        for param, value in model.best_params_.items():
            print(f"    - {param}: {value}")
        print("-" * 60)

# ==================== OUTPUT HASIL TUNING ====================
# Membuat dictionary model yang sudah dituning
tuned_models = {
    "Decision Tree": dt_bayes,
    "Random Forest": rf_bayes,
    "Naive Bayes": nb_bayes,
    "XGBoost": xgb_bayes
}

# Cetak hasil
print_tuning_results(tuned_models)


# ## **Evaluation**

# ### **Evaluasi Metrik**

# In[75]:


# Fungsi untuk mengevaluasi model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return accuracy, precision, recall, f1

# Evaluasi semua model
models_tuned = {
    "Decision Tree (Tuned)": dt_bayes,
    "Random Forest (Tuned)": rf_bayes,
    "Naive Bayes (Tuned)": nb_bayes,
    "XGBoost (Tuned)": xgb_bayes
}

# Pastikan kita gunakan hasil PCA di test set (test_pca_df) dan target yang sudah di-encode (new_y_test)
results = []
for name, model in models_tuned.items():
    acc, prec, rec, f1 = evaluate_model(model, test_pca_df, new_y_test)
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

# Tampilkan hasil dalam tabel
results_df = pd.DataFrame(results)
print("\n📌 **Model Evaluation Results After Tuning**:\n")
print(results_df.to_markdown(index=False))


# ### **Classification Reports**

# In[76]:


def print_classification_reports(models_dict, X_test, y_test):
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        
        # Format support as integer
        report_df['support'] = report_df['support'].apply(lambda x: int(x) if isinstance(x, (int, float)) else x)
        
        print(f"\n📌 **Classification Report - {name}**:\n")
        print(report_df[['precision', 'recall', 'f1-score', 'support']].to_markdown(floatfmt=".6f"))
print("\n📌 **Classification Report After Tuning:**\n")
print_classification_reports(models_tuned, test_pca_df, new_y_test)


# ### **Confussion Matrix**

# In[77]:


def plot_confusion_matrices(models_dict, X_test, y_test, class_names):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_test)
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f"{name} - Confusion Matrix")
    
    plt.tight_layout()
    plt.show()

# Ambil nama kelas dari encoder
class_names = list(encoder.classes_)

# Plot confusion matrix
plot_confusion_matrices(models_tuned, test_pca_df, new_y_test, class_names)


# ### **Feature Importance**

# In[78]:


def plot_feature_importance(model, feature_names, top_n=20):
    # Ambil nilai importance dari model
    feature_importance = model.best_estimator_.feature_importances_
    
    # Buat DataFrame untuk menampilkan secara terurut
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values(by='Importance', ascending=False)

    # Tampilkan tabel secara rapi
    print("\n📌 **Feature Importance berdasarkan Random Forest (Top 20)**:")
    display(feature_importance_df.head(top_n))

    # Plot
    plt.figure(figsize=(8, 6))
    sns.barplot(y='Feature', x='Importance', data=feature_importance_df.head(top_n), palette='Blues_r')
    plt.title('Top 20 Feature Importance - Random Forest')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

# Panggil fungsi
plot_feature_importance(rf_bayes, feature_names=train_pca_df.columns, top_n=20)


# Fungsi untuk menampilkan feature importance dari model Random Forest
# secara terurut (descending).
# 
# Parameters:
# - model: Model yang sudah di-fit (contohnya rf_bayes)
# - feature_names: List nama fitur yang digunakan
# - top_n: Jumlah fitur teratas yang akan ditampilkan (default 20)

# ## **Save Best Model**

# In[79]:


joblib.dump(rf_bayes, "model/randomForrest_model.joblib")


# ## **Requirements Dependencies**

# In[80]:


get_ipython().system('pip freeze > requirements.txt')

