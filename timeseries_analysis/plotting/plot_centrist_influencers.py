import shared_functions as sf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def fetch_num_narratives(p_a, p_b, year=None):
    if year:
        file_loc = f"../cleaned_data_end_april_monthbins/{year}/{p_a}_combo_hero, villain, victim_{p_b}.csv"
    else:
        file_loc = f"../cleaned_data_end_april_monthbins/{p_a}_combo_hero, villain, victim_{p_b}.csv"

    data = sf.import_csv(file_loc)
    return len(data)

def plot_center_as_influenced():
    records = []
    for part_a in sf.PARTISANSHIPS:
        for part_b in [x for x in sf.PARTISANSHIPS if 'Center' in x]:
            if part_a ==part_b:
                continue
            num_narratives = fetch_num_narratives(part_a, part_b)
            records.append({'Influencer':part_a,
                            'Influenced':part_b,
                            'Number of Narratives':num_narratives})

    df = pd.DataFrame(records)

    sns.barplot(x='Influencer', y='Number of Narratives', hue='Influenced', data=df)
    plt.title(' Bar Plot of the Number of Narratives in which the other Partisanships Influence the Center')
    plt.show()

def plot_center_as_influenced_by_year():
    for year in range(2016, 2023):
        records = []
        for part_a in sf.PARTISANSHIPS:
            for part_b in [x for x in sf.PARTISANSHIPS if 'Center' in x]:
                if part_a == part_b:
                    continue
                num_narratives = fetch_num_narratives(part_a, part_b, year)
                records.append({'Influencer': part_a,
                                'Influenced': part_b,
                                'Number of Narratives': num_narratives})

        df = pd.DataFrame(records)

        sns.barplot(x='Influencer', y='Number of Narratives', hue='Influenced', data=df)
        plt.title(
            f'{year}: Bar Plot of the Number of Narratives in which the other Partisanships Influence the Center')
        plt.show()


def plot_center_as_influencer():
    records = []
    for part_a in [x for x in sf.PARTISANSHIPS if 'Center' in x]:
        for part_b in sf.PARTISANSHIPS:
            if part_a == part_b:
                continue
            num_narratives = fetch_num_narratives(part_a, part_b)
            records.append({'Influencer': part_a,
                            'Influenced': part_b,
                            'Number of Narratives': num_narratives})

    df = pd.DataFrame(records)

    sns.barplot(x='Influencer', y='Number of Narratives', hue='Influenced', data=df)
    plt.title(
        'Bar Plot of the Number of Narratives in which the Center Influences other Partisanships')
    plt.show()


def plot_center_as_influencer_by_year():
    for year in range(2016, 2023):
        records = []
        for part_a in [x for x in sf.PARTISANSHIPS if 'Center' in x]:
            for part_b in sf.PARTISANSHIPS:
                if part_a == part_b:
                    continue
                num_narratives = fetch_num_narratives(part_a, part_b, year)
                records.append({'Influencer': part_a,
                                'Influenced': part_b,
                                'Number of Narratives': num_narratives})

        df = pd.DataFrame(records)

        sns.barplot(x='Influencer', y='Number of Narratives', hue='Influenced', data=df)
        plt.title(
            f'{year}: Bar Plot of the Number of Narratives in which the Center Influences other Partisanships')
        plt.show()


def plot_center_as_influenced_over_time():
    records = []
    for year in range(2016, 2023):

        for part_a in sf.PARTISANSHIPS:
            for part_b in [x for x in sf.PARTISANSHIPS if 'Center' in x]:
                if part_a == part_b:
                    continue
                num_narratives = fetch_num_narratives(part_a, part_b, year)
                records.append({'Influencer': part_a,
                                'Influenced': part_b,
                                'Number of Narratives': num_narratives,
                                'Year': year})

    df = pd.DataFrame(records)


    # 1. Line Plot
    gradient_colors = [
        '#000000',
        '#131313',
        '#262626',
        '#393939',
        '#4d4d4d',
        '#606060',
        '#737373',
        '#868686',
        '#9a9a9a',
        '#adadad',
        '#c0c0c0',
        '#d3d3d3'
    ]

    colors = [
        '#FF5733',  # Vibrant Orange
        '#33FF57',  # Bright Green
        '#3357FF',  # Bold Blue
        '#FF33A1',  # Hot Pink
        '#FFC300',  # Golden Yellow
        '#DAF7A6',  # Light Green
        '#900C3F',  # Deep Maroon
        '#581845',  # Dark Purple
        '#C70039'  # Crimson Red
    ]
    c_counter, g_counter = 0, 0
    for pair in df.groupby(['Influencer', 'Influenced']):
        print(c_counter, g_counter)
        subset = pair[1]
        label = f"{list(pair[0])[0]} -> {list(pair[0])[1]}"
        if label.count('Center') == 2:
            linestyle = '-'
            color = colors[c_counter]
            c_counter += 1
        else:
            linestyle = '--'
            color = gradient_colors[g_counter]
            g_counter += 1
        plt.plot(subset['Year'].to_list(), subset['Number of Narratives'].to_list(),
                 marker='o',
                 label=label,
                 color=color,
                 linestyle=linestyle)
    plt.title('Influence Strength Over Time')
    plt.xlabel('Year')
    plt.ylabel( 'Number of Narratives')
    plt.legend()
    plt.show()

    # 2. Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Year', y= 'Number of Narratives', hue='Influencer', data=df)
    plt.title('Influence Strength by Year')
    plt.show()

    # 3. Heatmap
    pivot_table = df.pivot_table(values= 'Number of Narratives', index=['Influencer', 'Influenced'], columns='Year')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
    plt.title('Heatmap of Influence Strength Over Years')
    plt.show()

def plot_center_as_influencer_over_time():
    records = []
    for year in range(2016, 2023):

        for part_a in [x for x in sf.PARTISANSHIPS if 'Center' in x]:
            for part_b in sf.PARTISANSHIPS:
                if part_a == part_b:
                    continue
                num_narratives = fetch_num_narratives(part_a, part_b, year)
                records.append({'Influencer': part_a,
                                'Influenced': part_b,
                                'Number of Narratives': num_narratives,
                                'Year': year})

    df = pd.DataFrame(records)


    # 1. Line Plot
    plt.figure(figsize=(10, 6))
    gradient_colors = [
        '#000000',
        '#131313',
        '#262626',
        '#393939',
        '#4d4d4d',
        '#606060',
        '#737373',
        '#868686',
        '#9a9a9a',
        '#adadad',
        '#c0c0c0',
        '#d3d3d3'
    ]

    colors = [
        '#FF5733',  # Vibrant Orange
        '#33FF57',  # Bright Green
        '#3357FF',  # Bold Blue
        '#FF33A1',  # Hot Pink
        '#FFC300',  # Golden Yellow
        '#DAF7A6',  # Light Green
        '#900C3F',  # Deep Maroon
        '#581845',  # Dark Purple
        '#C70039'  # Crimson Red
    ]
    c_counter, g_counter = 0,0
    for pair in df.groupby(['Influencer', 'Influenced']):
        print(c_counter,g_counter)
        subset = pair[1]
        label = f"{list(pair[0])[0]} -> {list(pair[0])[1]}"
        if label.count('Center')==2:
            linestyle='-'
            color=colors[c_counter]
            c_counter+=1
        else:
            linestyle='--'
            color = gradient_colors[g_counter]
            g_counter+=1
        plt.plot(subset['Year'].to_list(), subset[ 'Number of Narratives'].to_list(),
                 marker='o',
                 label=label,
                 color=color,
                 linestyle=linestyle)

    plt.title('Influence Strength Over Time')
    plt.xlabel('Year')
    plt.ylabel( 'Number of Narratives')
    plt.legend()
    plt.show()

    # 2. Bar Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Year', y= 'Number of Narratives', hue='Influencer', data=df)
    plt.title('Influence Strength by Year')
    plt.show()

    # 3. Heatmap
    pivot_table = df.pivot_table(values= 'Number of Narratives', index=['Influencer', 'Influenced'], columns='Year')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, cmap='coolwarm')
    plt.title('Heatmap of Influence Strength Over Years')
    plt.show()


if __name__ == '__main__':
    plot_center_as_influencer_over_time()
    plot_center_as_influenced_over_time()
    plot_center_as_influenced()
    # plot_center_as_influenced_by_year()

    plot_center_as_influencer()
    # plot_center_as_influencer_by_year()

    # TODO UPDATE FINDINGS DOC WITH THIS^ PLOT