from joblib import Parallel, delayed

def get_clusters_ids(clusters):
    return list(clusters.keys())

def get_clustured_ids(clusters):
    clustered_ids = set(
        [transaction[0] for cluster in clusters.values() for transaction in cluster]
    )
    clustered_ids |= set(clusters.keys())
    return clustered_ids

def get_unclustured_ids(ids, clusters):
    clustered_ids = get_clustured_ids(clusters)
    unclustered_ids = list(set(ids) - clustered_ids)
    return unclustered_ids

def nearest_cluster(
    transaction_ids,
    embeddings,
    clusters=None,
    parallel=None,
    threshold=0.75,
    chunk_size=2500,
):
    cluster_ids = list(clusters.keys())
    if len(cluster_ids) == 0:
        return clusters
    cluster_embeddings = get_embeddings(cluster_ids, embeddings)

    c = list(chunk(transaction_ids, chunk_size))

    with log_durations(logging.info, "Parallel jobs nearest cluster"):
        out = parallel(
            delayed(nearest_cluster_chunk)(
                chunk_ids,
                get_embeddings(chunk_ids, embeddings),
                cluster_ids,
                cluster_embeddings,
                threshold,
            )
            for chunk_ids in tqdm(c)
        )
        cluster_assignment = [assignment for sublist in out for assignment in sublist]

    for (transaction_id, similarity), cluster_id in cluster_assignment:
        if cluster_id is None:
            continue
        clusters[cluster_id].append(
            (transaction_id, similarity)
        )  # TODO sort in right order

    clusters = {
        cluster_id: unique_txs(sort_cluster(cluster))
        for cluster_id, cluster in clusters.items()
    }  # Sort based on similarity

    return clusters

def online_community_detection(ids,embeddings,clusters=None,threshold=0.7,min_cluster_size=3,chunk_size=2500,
                               iterations=10,cores=1):
    if clusters is None:
        clusters = {}

    with Parallel(n_jobs=cores) as parallel:
        for iteration in range(iterations):
            print("1. Nearest cluster")
            unclustered_ids = get_unclustured_ids(ids, clusters)
            cluster_ids = list(clusters.keys())
            print("Unclustured", len(unclustered_ids))
            print("Clusters", len(cluster_ids))
            clusters = nearest_cluster(
                unclustered_ids,
                embeddings,
                clusters,
                chunk_size=chunk_size,
                parallel=parallel,
            )
            print("\n\n")

            print("2. Create new clusters")
            unclustered_ids = get_unclustured_ids(ids, clusters)
            print("Unclustured", len(unclustered_ids))
            new_clusters = create_clusters(
                unclustered_ids,
                embeddings,
                clusters={},
                min_cluster_size=3,
                chunk_size=chunk_size,
                threshold=threshold,
                parallel=parallel,
            )
            new_cluster_ids = list(new_clusters.keys())
            print("\n\n")

            print("3. Merge new clusters", len(new_cluster_ids))
            max_clusters_size = 25000
            while True:
                new_cluster_ids = list(new_clusters.keys())
                old_new_cluster_ids = new_cluster_ids
                new_clusters = create_clusters(
                    new_cluster_ids,
                    embeddings,
                    new_clusters,
                    min_cluster_size=1,
                    chunk_size=max_clusters_size,
                    threshold=threshold,
                    parallel=parallel,
                )
                new_clusters = filter_clusters(new_clusters, 2)

                new_cluster_ids = list(new_clusters.keys())
                print("New merged clusters", len(new_cluster_ids))
                if len(old_new_cluster_ids) < max_clusters_size:
                    break

            new_clusters = filter_clusters(new_clusters, min_cluster_size)
            print(
                f"New clusters with min community size >= {min_cluster_size}",
                len(new_clusters),
            )
            clusters = {**new_clusters, **clusters}
            print("Total clusters", len(clusters))
            clusters = sort_clusters(clusters)
            print("\n\n")

            print("4. Nearest cluster")
            unclustered_ids = get_unclustured_ids(ids, clusters)
            cluster_ids = list(clusters.keys())
            print("Unclustured", len(unclustered_ids))
            print("Clusters", len(cluster_ids))
            clusters = nearest_cluster(
                unclustered_ids,
                embeddings,
                clusters,
                chunk_size=chunk_size,
                parallel=parallel,
            )
            clusters = sort_clusters(clusters)

            unclustered_ids = get_unclustured_ids(ids, clusters)
            clustured_ids = get_clustured_ids(clusters)
            print("Clustured", len(clustured_ids))
            print("Unclustured", len(unclustered_ids))
            print(
                f"Percentage clustured {len(clustured_ids) / (len(clustured_ids) + len(unclustered_ids)) * 100:.2f}%"
            )

            print("\n\n")
    return clusters


clusters = online_community_detection(ids, embeddings, clusters, chunk_size=5000)
