import matplotlib.pyplot as plt
import shared_functions as sf
import pandas as pd


def extract_values(dictionary):
    return dictionary['$oid']


def import_data():
    files = sf.get_files_from_folder('sampled_pooled_alphabetized2', 'json')

    counter = {f"{l}/{j}": 0 for l in range(1, 13) for j in range(2016, 2023)}

    for file in files:
        data = sf.import_json(file)
        df = pd.DataFrame(data)


        for index, row in df.iterrows():
            datetime_obj = pd.to_datetime(row.iloc[5])
            year = datetime_obj.year
            month = datetime_obj.month
            counter[f"{month}/{year}"] += 1
    x,y = [],[]
    for yr in range(2016, 2023):
        for m in range(1,13):
            y.append(counter[f"{m}/{yr}"])
            x.append(f"{m}/{yr}")


    return x,y

x, y = import_data()
print(sum(y))
plt.bar(x,y)
plt.xticks(rotation=90)
plt.xlabel('Time')
plt.ylabel('Number of articles')
plt.show()
