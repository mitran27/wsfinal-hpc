IDs only, no MS MARCO text (MS MARCO is licensed for non-commercial research; its text is downloaded from the
official site by jobs/setup_env.sh and checked against collection_subset.sha256).
- subset_pids.txt.gz: passage ids of the 1M-passage course subset
- qrels.*.tsv: course dev qrels and TREC DL 2019/2020 qrels (query id, passage id, grade)
- work.tgz: BM25 candidate pools, query splits and BM25 runs (ids, ranks and scores)
