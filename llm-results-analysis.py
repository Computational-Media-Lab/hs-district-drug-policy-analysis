import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Read in results files
df_district_name = pd.read_csv('FILE_PATH') # Redacted due to data privacy


# Combine all dataframes into one
df_list = ["List of all district dataframes"]
df = pd.concat(df_list, ignore_index = True)


# Create two new summary columns for disciplinary & health/treatment terms mentioned
df['total number of disciplinary terms mentioned'] = (
    df['number of times suspension or synonyms are mentioned'] +
    df['number of times expulsion or synonyms is mentioned'] +
    df['number of time transfer or referral to disciplinary alternative school is mentioned'] +
    df['number of times transfer or referral to juvenile justice program is mentioned'] + 
    df['number of times arrest or synonyms are mentioned'] + 
    df['number of times drug sniffing dogs or other drug screening procedures are mentioned'] +
    df['number of times police or school resource officers are mentioned'] +
    df['number of times any other terms are mentioned that relate to disciplinary approaches to student drug and alcohol use']
)

df['total number of health or treatment terms mentioned'] = (
    df['number of times transfer or referral to treatment programs are mentioned'] +
    df['number of times transfer or referral to recovery schools are mentioned'] +
    df['number of times school based health centers or synonyms are mentioned'] + 
    df['number of times transfer or referral to mental health clinics or other off campus health facilities are mentioned'] + 
    df['number of times trauma-informed care or synonyms are mentioned'] + 
    df['number of times restorative justice approaches or synonyms are mentioned'] + 
    df['number of times drug and alcohol use prevention programs are mentioned'] + 
    df['number of times health-services staff such as school nurses, social workers, psychologists, or substance use counselors are mentioned'] +
    df['number of times any other terms are mentioned that relate to a health-oriented approach to student drug and alcohol use']
)

col1 = df.pop("total number of disciplinary terms mentioned")
col2 = df.pop("total number of health or treatment terms mentioned")

df.insert(3, col1.name, col1)
df.insert(4, col2.name, col2)


df['d_ratio'] = df['total number of disciplinary terms mentioned'] / (df['total number of disciplinary terms mentioned'] + df['total number of health or treatment terms mentioned'])
df['h_ratio'] = df['total number of health or treatment terms mentioned'] / (df['total number of disciplinary terms mentioned'] + df['total number of health or treatment terms mentioned'])

# Create alias column for districts and drop original district column name
district_type_map = {"District Names": "District Aliases"} # Redacted for data privacy purposes

df['district-alias'] = df['district'].map(district_type_map)
df = df.drop(columns = ['district', 'state'])

df_d_ratio = df.sort_values(by = 'd_ratio').reset_index(drop = True)

fig, ax = plt.subplots(figsize = (12, 5), dpi = 150)

ax.axhline(0, color = '#cccccc', linestyle = '--', linewidth = 1.5, zorder = 1)
ax.scatter(df['d_ratio'], np.zeros(len(df)), color = '#2b5c8f', s = 120, zorder = 3)

offsets = [20, -30, 45, -65, 85, -30, 35, -45, 70, 50, 35, -45]

for i, row in df.iterrows():
    offset = offsets[i % len(offsets)]
    ax.annotate(
        row['district-alias'],
        (row['d_ratio'], 0),
        xytext=(0, offset),
        textcoords="offset points",
        ha='center',
        va='bottom' if offset > 0 else 'top',
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8)
    )

ax.set_ylim(-0.1, 0.1)
ax.get_yaxis().set_visible(False) 
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Disciplinary Terms Ratio', fontsize = 12, labelpad = 10)
ax.set_title('District Disciplinary Terms Index Spectrum', fontsize = 14, pad = 10)
plt.tight_layout()
plt.show()


df_d_ratio = df.sort_values(by = 'h_ratio').reset_index(drop = True)

fig, ax = plt.subplots(figsize = (12, 5), dpi = 150)

ax.axhline(0, color = '#cccccc', linestyle = '--', linewidth = 1.5, zorder = 1)
ax.scatter(df['h_ratio'], np.zeros(len(df)), color = '#2b5c8f', s = 120, zorder = 3)

offsets = [20, -30, 45, -65, 85, -30, 35, -45, 70, 50, 35, -45]

for i, row in df.iterrows():
    offset = offsets[i % len(offsets)]
    ax.annotate(
        row['district-alias'],
        (row['h_ratio'], 0),
        xytext=(0, offset),
        textcoords="offset points",
        ha='center',
        va='bottom' if offset > 0 else 'top',
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.8)
    )

ax.set_ylim(-0.1, 0.1)
ax.get_yaxis().set_visible(False) 
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xlabel('Health Terms Ratio', fontsize = 12, labelpad = 10)
ax.set_title('District Health Terms Index Spectrum', fontsize = 14, pad = 10)
plt.tight_layout()
plt.show()