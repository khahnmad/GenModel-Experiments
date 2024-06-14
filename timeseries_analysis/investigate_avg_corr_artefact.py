import shared_functions as sf
import pandas as pd
import datetime
import matplotlib.pyplot as plt
# Plot entire data as histogram with 15 day bins
# x axis bin, y axis frequency

def apply_binning(elts: list) -> list:
    date_ids = {}
    count = 0
    for y in range(2016, 2023):
        for m in range(1, 13):
            for half in [1, 15]:
                date_ids[datetime.date(year=y, month=m, day=half)] = count
            count += 1

    # Given a list of data points, caputre the date info and put them into the appropriate bins
    binned = []
    for elt in elts:
        date = elt[5]
        if date == 'ERROR: No publish date':
            continue
        datetime_obj = pd.to_datetime(date)
        if datetime_obj is None:
            continue

        month = datetime_obj.month
        year = datetime_obj.year

        if datetime_obj.day <= 15:
            month_half = 1
        else:
            month_half = 15
        doc = {'hero': elt[0],
               'villain': elt[1],
               'victim': elt[2],
               'combo': f"{elt[0]}.{elt[1]}.{elt[2]}",
               'month': month,
               'year': year,
               'month_half': month_half,
               'time': datetime.date(year=year, month=month, day=month_half),
               'partisanship': elt[4],
               'media_name': elt[6] if len(elt) > 6 else None,
               'date_id': date_ids[ datetime.date(year=year, month=month, day=month_half)]}
        binned.append(doc)

    return binned

def plot_signal_histogram(level, h_v_v):
    # for source in ['FarRight','Right','CenterRight','Center','CenterLeft','Left','FarLeft']:
    #     output = sf.import_json(f'signals/{source}_signals_by_part_input_hvv.json')[source]
    #     for p_b in ['FarRight', 'Right', 'CenterRight', 'Center', 'CenterLeft', 'Left', 'FarLeft']:
    #         if p_b == source:
    #             continue
    #         rel_signals = [x[2] for x in output[p_b][level][h_v_v]]
    #         df = pd.DataFrame(rel_signals)
    #         col_summary = df[df.columns].sum().to_list()
    #         print('')
    pool_alphabet_files = sf.get_files_from_folder('sampled_pooled_alphabetized2', 'json')
    hvv_indexing = {'hero': 0, 'villain': 1, 'victim': 2}

    content = []
    for source in ['FarRight', 'Right', 'CenterRight', 'Center', 'CenterLeft', 'Left', 'FarLeft']:
        file = [x for x in pool_alphabet_files if f'\\{source}_data' in x][0]

        data = sf.import_json(file)
        binned_objs = apply_binning(data)

        content+= binned_objs

        part_df = pd.DataFrame(binned_objs)
        part_df = part_df.drop_duplicates()  # remove any dupliactes, ie any time/partisanship combinations where the same

        ## Difference
        counted = part_df[['date_id','month_half']].value_counts().reset_index().sort_values(by='date_id')
        y = []
        for i in counted['date_id'].values:
            try:
                a = list(counted[counted['date_id'] == i][counted['month_half'] == 1]['count'].values)[0]
            except IndexError:
                a= 0
            try:
                b = list(counted[counted['date_id'] == i][counted['month_half'] == 15]['count'].values)[0]
            except IndexError:
                b = 0
            difference = a-b
            y.append(difference)
        x = list(counted['date_id'].values)

        plt.bar(x, y)
        plt.title(f'{source}, Difference Between Number of Articles in Each Half-Month Bin')
        plt.xlabel('Month')
        plt.ylabel('Number of Articles Difference')
        plt.show()
        # counted = part_df['date_id'].value_counts().sort_index()
        # x = counted.index.to_list()
        # y = list(counted.values)
        # plt.bar(x, y)
        # plt.title(f'{source}, Number of Articles in Each Half-Month Bin')
        # plt.xlabel('Half-Month Bin')
        # plt.ylabel('Number of Articles')
        # plt.show()


    # Convert to signal
    # binned_objs = apply_binning(content)
    df = pd.DataFrame(content)  # reformat

    df = df.drop_duplicates()  # remove any dupliactes, ie any time/partisanship combinations where the same
    counted = df['date_id'].value_counts().sort_index()
    x = counted.index.to_list()
    y = list(counted.values)
    plt.bar(x,y)
    plt.title('Entire Dataset, Number of Articles in Each Half-Month Bin')
    plt.xlabel('Half-Month Bin')
    plt.ylabel('Number of Articles')
    plt.show()

def plot_month_time_distribution(source, year):
    pool_alphabet_files = sf.get_files_from_folder('sampled_pooled_alphabetized2', 'json')

    file = [x for x in pool_alphabet_files if f'\\{source}_data' in x][0]
    data = sf.import_json(file)
    days = []
    for elt in data:
        date = elt[5]
        if date == 'ERROR: No publish date':
            continue
        datetime_obj = pd.to_datetime(date)
        if datetime_obj is None:
            continue
        if year != datetime_obj.year:
            continue
        days.append(datetime_obj.day)
    df = pd.DataFrame(days)
    value_counts = df.value_counts().reset_index().sort_values(by=0)
    x = list(value_counts[0].values)
    y = list(value_counts['count'].values)
    plt.bar(x, y)
    plt.title(f'{source}, {year} Frequency of Articles from Each Day of the Month')
    plt.xlabel('Day of the Month')
    plt.ylabel('Number of Articles')
    plt.show()
    print('')

inputs = {
    'single':['hero','villain','victim'],
    'combo': ['hero, villain, victim'],
    'tuple':[['hero','villain'],['hero','victim'],['villain','victim']]
}

# for input_level in inputs.keys():
#     for hvv in inputs[input_level]:
#         plot_signal_histogram(input_level, hvv)

for part in ['FarRight', 'Right', 'CenterRight', 'Center', 'CenterLeft', 'Left', 'FarLeft']:
    for year in range(2016,2023):
        plot_month_time_distribution(part, year)