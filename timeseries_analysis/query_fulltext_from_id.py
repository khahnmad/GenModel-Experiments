import shared_functions as sf
from bson import ObjectId

data = sf.import_json('matched_article_ids.json')['content']
narr_matches = sf.import_json('narrative_matches_months.json')
# lets start with far right far left narr matches,
fr_fl = ['FarRight','FarLeft']
narratives =list(set( [x['narrative'] for x in narr_matches if x['first_partisanship'] in fr_fl and x['lagging_partisanship'] in fr_fl
                       and x['first_partisanship']!=x['lagging_partisanship']
                       and x['hvv_combination_type']=='combo']))
#group by narrative
# narratives = list(set([f"{x[0]}.{x[1]}.{x[2]}" for x in data]))

fs,db = sf.getConnection()

for narr in narratives:

    id_list = [ObjectId(x[3]) for x in data if f"{x[0]}.{x[1]}.{x[2]}"==narr]
    docs = sf.query_collection(collection=db.articles.sampled.triplets,
                               query={"_id": {"$in": id_list}},
                               fields=['parsing_result','partisanship']
                               )
    rel_docs = [x for x in docs if x['partisanship'] in fr_fl]
    if len(rel_docs)>0:
        print(narr)
    # for d in rel_docs:
    #     print(f"{d['_id']}, {d['partisanship']}")
    print()