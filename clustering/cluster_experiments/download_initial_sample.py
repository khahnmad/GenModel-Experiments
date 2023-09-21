import shared_functions as sf

############################################# FULL SUBSAMPLE ####################################################
fs,db = sf.getConnection(use_dotenv=True)

# Query for vectors
docs = sf.query_collection(collection=db['sampled_articles'],
                           query={'initial_subsample':True},
                           fields=['processing_result','denoising_result','embedding_result','sample_id']
                           )

# Export sample
sf.export_as_json('input/initial_subsample_results.json', docs)
print(f'COMPLETE, exported {len(docs)}')


############################################# INITIAL TEST ####################################################
# # Open connection
# fs,db = sf.getConnection(use_dotenv=True)
#
# # Query for vectors
# docs = sf.query_collection(collection=db['sampled_articles'],
#                            query={'initial_subsample':True},
#                            fields=['processing_result','denoising_result','embedding_result','sample_id'],
#                            n=100)
#
# # Export sample
# sf.export_as_json('100_sampled_articles.json',docs)