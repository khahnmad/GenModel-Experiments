import shared_functions as sf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def fetch_data():
    pool_alphabet_files = sf.get_files_from_folder('../sampled_pooled_alphabetized2', 'json')
    records = []
    for file in pool_alphabet_files:
        data = sf.import_json(file)
        for row in data:
            records.append({'Year':int(row[5][:4]),
                            'Partisanship':row[4]})
    df = pd.DataFrame(records)
    grouped = df.groupby(by=['Year', 'Partisanship']).size().reset_index()
    grouped.columns = ['Year', 'Partisanship','Count']
    return grouped

def plot_all(df):
    partisanship = df['Partisanship'].values.tolist()
    count = df['Count'].values.tolist()
    year = df['Year'].values.tolist()
    # Line plot
    sns.lineplot(x=year, y=count, hue=partisanship, marker='o')
    plt.title('Partisanship Count Over Time')
    plt.show()
    # Bar plot
    sns.barplot(x=year, y=count, hue=partisanship)
    plt.title('Partisanship Count by Year')
    plt.show()
    # Pivot the DataFrame for stacked bar plot
    pivot_df = df.pivot(index=year, columns=partisanship, values=count)

    # Stacked bar plot
    pivot_df.plot(kind='bar', stacked=True)
    plt.title('Stacked Bar Plot of Partisanship Counts by Year')
    plt.ylabel('Count')
    plt.show()
    # Area plot
    pivot_df.plot(kind='area', stacked=True)
    plt.title('Area Plot of Partisanship Counts Over Time')
    plt.ylabel('Count')
    plt.show()
    # Box plot
    sns.boxplot(x='Year', y='Count', hue='Partisanship', data=df)
    plt.title('Box Plot of Partisanship Counts by Year')
    plt.show()
    # Group by Year and sum the counts
    total_change = df.groupby('Year')['Count'].sum().reset_index()

    # Line plot of the total change over time
    plt.plot(total_change['Year'], total_change['Count'], marker='o')
    plt.title('Total Partisanship Count Change Over Time')
    plt.xlabel('Year')
    plt.ylabel('Total Count')
    plt.grid(True)
    plt.show()
    # Bar plot of the total change over time
    plt.bar(total_change['Year'], total_change['Count'], color='skyblue')
    plt.title('Total Partisanship Count Change Over Time')
    plt.xlabel('Year')
    plt.ylabel('Total Count')
    plt.show()
    return

if __name__ == '__main__':
    data = fetch_data()
    plot_all(data)