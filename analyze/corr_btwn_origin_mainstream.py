import shared_functions as sf
import dateutil.parser

x = [] # narrative origin
y = [] # mainstream acceptance, ie % in centrist narratives

"""
Ways to define y
- % in centrist publications 
- % of other partisanships in which it appears 
- total num appearances outside of its original partisanhip
- """
import matplotlib.pyplot as plt
data = sf.import_json('sample_hvv_appearances.json')

for narr in data:
    oldest_date = min([dateutil.parser.parse(x[1]) for x in data[narr]])
    origin_part = [x[0] for x in data[narr] if dateutil.parser.parse(x[1])== oldest_date][0]
    perc_centrist = len([x[0] for x in data[narr] if "Center" in x[0]])/len(data[narr])
    x.append(origin_part)
    y.append(perc_centrist)

plt.scatter(x,y)
plt.show()