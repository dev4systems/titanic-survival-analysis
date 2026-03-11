import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project paths (works from project root or src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "train.csv"
VIS_DIR = PROJECT_ROOT / "visualizations"
VIS_DIR.mkdir(exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)

# Part A: Inspect
print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

# Part B: Univariate Analysis
print("\n" + "="*50)
print("PART B: UNIVARIATE ANALYSIS")
print("="*50)

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# 1. Survival Distribution
plt.subplot(2, 3, 1)
df['Survived'].value_counts().plot(kind='bar', color=['#d62728', '#2ca02c'])
plt.title('Survival Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Count')
plt.xticks(rotation=0)

# 2. Passenger Class Distribution
plt.subplot(2, 3, 2)
df['Pclass'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Passenger Class Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Class (1=First, 2=Second, 3=Third)')
plt.ylabel('Count')
plt.xticks(rotation=0)

# 3. Sex Distribution
plt.subplot(2, 3, 3)
df['Sex'].value_counts().plot(kind='bar', color=['#1f77b4', '#ff7f0e'])
plt.title('Sex Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Sex')
plt.ylabel('Count')
plt.xticks(rotation=0)

# 4. Age Distribution
plt.subplot(2, 3, 4)
df['Age'].hist(bins=30, color='green', edgecolor='black', alpha=0.7)
plt.title('Age Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Age (years)')
plt.ylabel('Frequency')
plt.axvline(df['Age'].median(), color='red', linestyle='--', label=f'Median: {df["Age"].median():.1f}')
plt.legend()

# 5. Fare Distribution
plt.subplot(2, 3, 5)
df['Fare'].hist(bins=30, color='purple', edgecolor='black', alpha=0.7)
plt.title('Fare Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Fare ($)')
plt.ylabel('Frequency')
plt.axvline(df['Fare'].median(), color='red', linestyle='--', label=f'Median: {df["Fare"].median():.2f}')
plt.legend()

# 6. Embarked Distribution
plt.subplot(2, 3, 6)
df['Embarked'].value_counts().plot(kind='bar', color='teal')
plt.title('Port of Embarkation', fontsize=12, fontweight='bold')
plt.xlabel('Port (C=Cherbourg, Q=Queenstown, S=Southampton)')
plt.ylabel('Count')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig(VIS_DIR / '01_univariate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()  # Close instead of show() to continue execution

print("\n[OK] Univariate analysis visualization saved!")

# Part C: Bivariate Analysis (How features relate to Survival)
print("\n" + "="*50)
print("PART C: BIVARIATE ANALYSIS")
print("="*50)

plt.figure(figsize=(15, 10))

# 1. Sex vs Survival
plt.subplot(2, 3, 1)
sex_survival = pd.crosstab(df['Sex'], df['Survived'], normalize='index') * 100
sex_survival.plot(kind='bar', color=['#d62728', '#2ca02c'])
plt.title('Survival Rate by Sex (%)', fontsize=12, fontweight='bold')
plt.xlabel('Sex')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(['Died', 'Survived'])
for i, v in enumerate(sex_survival[1]):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# 2. Class vs Survival
plt.subplot(2, 3, 2)
class_survival = pd.crosstab(df['Pclass'], df['Survived'], normalize='index') * 100
class_survival.plot(kind='bar', color=['#d62728', '#2ca02c'])
plt.title('Survival Rate by Class (%)', fontsize=12, fontweight='bold')
plt.xlabel('Class (1=First, 2=Second, 3=Third)')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=0)
plt.legend(['Died', 'Survived'])
for i, v in enumerate(class_survival[1]):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center', fontweight='bold')

# 3. Age vs Survival (boxplot)
plt.subplot(2, 3, 3)
df.boxplot(column='Age', by='Survived', ax=plt.gca())
plt.title('Age Distribution by Survival', fontsize=12, fontweight='bold')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Age (years)')
plt.suptitle('')  # Remove automatic title

# 4. Fare vs Survival (boxplot)
plt.subplot(2, 3, 4)
df.boxplot(column='Fare', by='Survived', ax=plt.gca())
plt.title('Fare Distribution by Survival', fontsize=12, fontweight='bold')
plt.xlabel('Survived (0=No, 1=Yes)')
plt.ylabel('Fare ($)')
plt.suptitle('')

# 5. Embarked vs Survival
plt.subplot(2, 3, 5)
embarked_survival = pd.crosstab(df['Embarked'], df['Survived'], normalize='index') * 100
embarked_survival.plot(kind='bar', color=['#d62728', '#2ca02c'])
plt.title('Survival Rate by Port (%)', fontsize=12, fontweight='bold')
plt.xlabel('Port (C=Cherbourg, Q=Queenstown, S=Southampton)')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(['Died', 'Survived'])

# 6. Family Size vs Survival (SibSp + Parch)
plt.subplot(2, 3, 6)
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1 for the passenger themselves
family_survival = pd.crosstab(df['FamilySize'], df['Survived'], normalize='index') * 100
family_survival.plot(kind='bar', color=['#d62728', '#2ca02c'])
plt.title('Survival Rate by Family Size (%)', fontsize=12, fontweight='bold')
plt.xlabel('Family Size (including self)')
plt.ylabel('Percentage (%)')
plt.xticks(rotation=45)
plt.legend(['Died', 'Survived'])

plt.tight_layout()
plt.savefig(VIS_DIR / '02_bivariate_analysis.png', dpi=300, bbox_inches='tight')
plt.close()  # Close instead of show() to continue execution

print("\n[OK] Bivariate analysis visualization saved!")

# Print insights
print("\n" + "="*50)
print("KEY INSIGHTS FROM BIVARIATE ANALYSIS")
print("="*50)
print(f"\n1. Sex vs Survival:")
print(sex_survival)
print(f"\n2. Class vs Survival:")
print(class_survival)
print(f"\n3. Embarked vs Survival:")
print(embarked_survival)

# Part D: Missing Values Strategy & Feature Engineering Insights
print("\n" + "="*50)
print("PART D: MISSING VALUES STRATEGY")
print("="*50)

print("\nMissing Values Summary:")
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100
missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_percent})
print(missing_df[missing_df['Count'] > 0].sort_values('Count', ascending=False))

print("\n" + "="*50)
print("STRATEGY FOR EACH MISSING VALUE:")
print("="*50)

print("\n1. AGE (177 missing = 19.8%)")
print("   - Median Age:", df['Age'].median())
print("   - Strategy: FILL with median by Class + Sex")
print("   - Reason: Age likely varies by passenger class/gender")

print("\n2. CABIN (687 missing = 77.1%)")
print("   - Strategy: DROP column")
print("   - Reason: Too many missing values; can extract Deck from non-null values later")

print("\n3. EMBARKED (2 missing = 0.2%)")
print("   - Most common port:", df['Embarked'].mode()[0])
print("   - Strategy: FILL with mode (most frequent port)")
print("   - Reason: Only 2 missing; safe to fill with most common value")

print("\n" + "="*50)
print("FEATURE ENGINEERING INSIGHTS:")
print("="*50)

print("\nNew Features to Create:")
print("1. FamilySize = SibSp + Parch + 1")
print("   - Insight: Family size 1-4 had higher survival rates")
print("2. IsAlone = 1 if FamilySize == 1, else 0")
print("3. Title extraction from Name (Mr, Mrs, Miss, Master)")
print("   - Insight: Titles reveal gender/age which correlate with survival")
print("4. Deck extraction from Cabin (if available)")
print("5. Age groups: Child (< 18), Adult (18-60), Senior (> 60)")

print("\n" + "="*50)
print("DATA QUALITY ASSESSMENT:")
print("="*50)
print(f"\nTotal records: {len(df)}")
print(f"Survived: {df['Survived'].sum()} ({(df['Survived'].sum()/len(df)*100):.1f}%)")
print(f"Died: {(df['Survived']==0).sum()} ({((df['Survived']==0).sum()/len(df)*100):.1f}%)")
print(f"\nClass imbalance: {df['Survived'].value_counts().to_dict()}")
print("Recommendation: Use F1-Score, ROC-AUC (not just accuracy) for model evaluation")

print("\n" + "="*50)
print("[OK] EDA Complete! Ready for preprocessing and modeling")
print("="*50)