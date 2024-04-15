## just developing for the later
import matplotlib.pyplot as plt
import pandas as pd

def plot_data(data):
    df = pd.DataFrame(data)
    df['publish_date'] = pd.to_datetime(df['publish_date'])
    df['month'] = df['publish_date'].apply(lambda x : x.month) # create column with month values
    df['year'] = df['publish_date'].apply(lambda x : x.year) # create column with year values

    for p in ['FarRight', 'Right', 'CenterRight', 'Center', 'CenterLeft', 'Left', 'FarLeft']:
        x, x_ticks,y = [],[],[]
        count =0
        for yr in range(2016,2023):
            for m in range(1,13):
                x.append(count)
                x_ticks.append(f"{m}/{yr}")

                y.append(df[(df['partisanship']==p) & (df['month']==m) & (df['year']==yr)]['num_keywords'].sum())
                count +=1

        plt.plot(x,y, label=p)

    plt.xticks(ticks=x,labels=x_ticks)
    plt.legend()
    plt.show()
    print('')

keywords = {"Immigration":{"migrants":[{'publish_date': '2016-01-01', "partisanship":"FarRight", 'outlet':"BBC",
                                        "num_keywords":4},
                                       {'publish_date': '2016-01-15', "partisanship": "FarRight", 'outlet': "Fox",
                                        "num_keywords": 2},
                                       {'publish_date': '2016-07-01', "partisanship":"FarRight", 'outlet':"BBC",
                                        "num_keywords":1},
                                       {'publish_date': '2016-08-01', "partisanship":"FarRight", 'outlet':"Fox",
                                        "num_keywords":5}
                                       ]}}
topic = "Immigration"
keyword = "migrants"
plot_data(keywords[topic][keyword])