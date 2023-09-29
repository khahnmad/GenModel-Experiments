import shared_functions as sf

fs,db = sf.getConnection(use_dotenv=True)

# Query for vectors
docs = sf.query_collection(collection=db.articles.sampled.triplets,
                           query={"triplets": {"$exists": True}},
                           fields=['processing_result','denoising_result','embedding_result','publish_date','partisanship']
                           )

# Export sample
sf.export_as_json('initial_subsample_triplets_results.json', docs)
print(f'COMPLETE, exported {len(docs)}')

