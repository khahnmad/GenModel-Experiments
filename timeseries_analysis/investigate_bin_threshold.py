import shared_functions as sf
import pandas as pd
#
def extract_values(dictionary):
    oid = dictionary['$oid']

    return oid



# counter = {k:{j:{l: 0 for l in range(1,13)} for j in range(2016,2023)} for k in ['FarRight', 'Right','CenterRight','Center','CenterLeft','Left','FarLeft']}
# # Iterate through the pooled files
# pool_alphabet_files = sf.get_files_from_folder('pooled_alphabetized2', 'json')
# for file in pool_alphabet_files:
#     data = sf.import_json(file)['content']
#     df = pd.DataFrame(data)
#     df[3] =df[3].apply(lambda x: pd.Series(extract_values(x)))
#     df = df.drop_duplicates()
#     for index, row in df.iterrows():
#         part = row.iloc[4]
#         datetime_obj = pd.to_datetime(row.iloc[5])
#         if datetime_obj is None:
#             continue
#         year = datetime_obj.year
#         month = datetime_obj.month
#         counter[part][year][month]+=1
#         # print('')
# # sf.export_as_json('sample_bin_size_no_dups.json',counter)
counter = sf.import_json('sample_bin_size_no_dups.json')
# # just the values
# values = []
# for p in counter.keys():
#     for yr in counter[p].keys():
#         values += list(counter[p][yr].values())
"""
X axis is bin size, y axis is the number of bins that meet that threshold 
"""
import matplotlib.pyplot as plt
def count_bins(data,threshold):
    count = 0
    for p in data.keys():
        for y in data[p].keys():
            for m in data[p][y].keys():
                if data[p][y][m] >= threshold:
                    count+=1
    return count

x_axis = list(range(100,590,10)) # set the possible thresholds

total_bins = len([f"{m}.{yr}" for yr in range(2016,2023) for m in range(1,13)])

# x_datapoints = x_axis
# y_datapoints = [count_bins(counter, t)*t for t in x_axis]

# Where is the 80% threshold?
eighty = 471
seventy = 412
sixty = 352
threehundred = 300
y_axis = []
for elt in x_axis:
    num = count_bins(counter, elt)
    y_axis.append(num)

plt.plot(x_axis, y_axis)
# plt.plot(x_datapoints, y_datapoints)
plt.plot([threehundred for i in range(len(x_axis))], y_axis)
# plt.plot([seventy for i in range(len(x_axis))], y_axis)
# plt.plot([sixty for i in range(len(x_axis))], y_axis)
plt.xlabel('Threshold for bin quantity')
plt.ylabel('Number of bins that meet the threshold')
plt.show()