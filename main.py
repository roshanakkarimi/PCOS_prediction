import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# import kagglehub
# Download latest version
# path = kagglehub.dataset_download("prasoonkottarathil/polycystic-ovary-syndrome-pcos")


# Data Preprocessing and cleaning
file_path_with_infertility = "data/PCOS_infertility.csv"
file_path_without_infertility = "data/PCOS_data_without_infertility.xlsx"
PCOS_winf = pd.read_csv(file_path_with_infertility)
PCOS_woinf = pd.read_excel(file_path_without_infertility, sheet_name="Full_new")

PCOS_winf.describe()
PCOS_woinf.describe()

#Mergin two datasets based on the Patient File No.
PCOS_data = pd.merge(PCOS_woinf, PCOS_winf, on="Patient File No.", suffixes=['','_w'],how='left')

#checking for duplicated columns
dup_cols = np.array([col for col in PCOS_data.columns if col.endswith("_w")])
drop_cols = np.append(dup_cols, "Unnamed: 44")

#drop duplicated cols and unnamed 44 (which is always null)
PCOS_data.drop(drop_cols, axis=1, inplace=True)

#Managinng Categorical values.
#In this database items with the object Dtype are just numeric values which saved as strings.
#we can just convert them into numeric values.
PCOS_data["AMH(ng/mL)"] = pd.to_numeric(PCOS_data["AMH(ng/mL)"], errors='coerce')
PCOS_data["II    beta-HCG(mIU/mL)"] = pd.to_numeric(PCOS_data["II    beta-HCG(mIU/mL)"], errors='coerce')

#Managing missing values.
#Filling NA values with the median of that feature.
PCOS_data.isnull().sum()

#Plotting box-plots for missed value attrs

#Marriage Status
sns.boxplot(x=PCOS_data['Marraige Status (Yrs)'], color='skyblue')
plt.title('Boxplot of Marriage Status (Years)')
plt.xlabel('Marriage Duration (Years)')
plt.show()

# II    beta-HCG(mIU/mL)
sns.boxplot(x=PCOS_data['II    beta-HCG(mIU/mL)'], color='skyblue')
plt.title('Boxplot of II    beta-HCG(mIU/mL)')
plt.xlabel('II    beta-HCG(mIU/mL)')
plt.show()

# AMH(ng/mL)
sns.boxplot(x=PCOS_data['AMH(ng/mL)'], color='skyblue')
plt.title('Boxplot of AMH(ng/mL)')
plt.xlabel('AMH(ng/mL)')
plt.show()

# based on the box-plots for each attribute i decided to choose wheter mean or media:
# if the data was skewed I'd choose median and otherwise mean,
# also mean is less chosen due to its sensitivity to outliers.

PCOS_data['Marraige Status (Yrs)'].fillna(PCOS_data['Marraige Status (Yrs)'].median(),inplace=True)
PCOS_data['II    beta-HCG(mIU/mL)'].fillna(PCOS_data['II    beta-HCG(mIU/mL)'].median(),inplace=True)
PCOS_data['AMH(ng/mL)'].fillna(PCOS_data['AMH(ng/mL)'].median(),inplace=True)
PCOS_data['Fast food (Y/N)'].fillna(PCOS_data['Fast food (Y/N)'].median(),inplace=True)


#Clearing up the extra space in the column names (optional)
PCOS_data.columns = [col.strip() for col in PCOS_data.columns]