import  pandas as pd
import shared_functions as sf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_apolitical_sex_misconduct_cases():
    data = sf.import_csv('tables/mainstream_cats.csv')
    cleaned = []
    for row in data[1:]:
        if row[2]=='hero':
            missing = np.mean([float(row[5]), float(row[6])])
        elif row[2]=='villain':
            missing = np.mean([float(row[4]), float(row[6])])
        elif row[2]=='victim':
            missing = np.mean([float(row[4]), float(row[5])])
        elif row[2]=='hero-villain':
            missing =row[6]
        elif row[2]=='hero-victim':
            missing =row[5]
        elif row[2]=='villain-victim':
            missing =row[4]
        row.append(float(missing))
        cleaned.append(row)

    df = pd.DataFrame(cleaned)
    df.columns=['index','character','hvv','partisanship','hero','villain','victim','category','Overlap Coefficient']
    # df.plot.scatter(x='partisanship',y='missing',c='category',
    #                     colormap='plasma')
    subset = df[(df['category']=='apolitical') | (df['category']=='sexual misconduct')]

    sns.relplot(data=subset, x='partisanship', y='Overlap Coefficient', hue='category', hue_order=list(set(subset['category'].values)))
    plt.title('Overlap between Remaining Character Archetypes')
    plt.tight_layout()
    plt.savefig('overlap_remaining_character_archetpyes.png')
    plt.show()

def plot_centrist_cases():
    data = sf.import_csv('tables/mainstream_cats.csv')
    cleaned = []
    for row in data[1:]:
        if 'Center' not in row[3]:
            continue
        if row[2] == 'hero':
            missing = np.mean([float(row[5]), float(row[6])])
        elif row[2] == 'villain':
            missing = np.mean([float(row[4]), float(row[6])])
        elif row[2] == 'victim':
            missing = np.mean([float(row[4]), float(row[5])])
        elif row[2] == 'hero-villain':
            missing = row[6]
        elif row[2] == 'hero-victim':
            missing = row[5]
        elif row[2] == 'villain-victim':
            missing = row[4]
        row.append(float(missing))
        cleaned.append(row)

    df = pd.DataFrame(cleaned)
    df.columns = ['index', 'character', 'hvv', 'partisanship', 'hero', 'villain', 'victim', 'category',
                  'Overlap Coefficient']
    # df.plot.scatter(x='partisanship',y='missing',c='category',
    #                     colormap='plasma')
    # subset = df[(df['category'] == 'apolitical') | (df['category'] == 'sexual misconduct')]

    sns.relplot(data=df, x='partisanship', y='Overlap Coefficient', hue='category',
                hue_order=list(set(df['category'].values)))
    plt.title('Overlap between Remaining Character Archetypes of Centrist Narratives')
    plt.tight_layout()
    plt.savefig('overlap_centrist_character_archetpyes.png')
    plt.show()

# plot_centrist_cases()
plot_apolitical_sex_misconduct_cases()