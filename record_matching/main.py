from .context import build_session_context, EMBEDDING_SIZE, MODEL
from vectorlink_py import template as tpl, dedup, embed
from vectorlink_py.utils import name_to_torch_type
import sys
import datafusion as df
from typing import Optional, Dict, Literal, List
import torch
from vectorlink_gpu.ann import ANN
from vectorlink_gpu.datafusion import dataframe_to_tensor, tensor_to_arrow
import pyarrow as pa
import pandas as pd
import numpy as np
import scipy
import re

from openai import OpenAI
import openai as oa

from sklearn.utils import shuffle
from sklearn.metrics.cluster import adjusted_rand_score

from sklearn import linear_model
from sklearn import metrics

"""
record: string
marcKey: string
person: string
roles: string
title: string
attribution: string
provision: string
subjects: string
genres: string
relatedWork: string
recordId: int64
"""

TEMPLATES = {
    "title": "{{#if title}}work title: {{title}}\n{{/if}}",
    "person": "{{#if person}}person name: {{person}}\n{{/if}}",
    "roles": "{{#if roles}}person roles: {{roles}}\n{{/if}}",
    "attribution": "{{#if attribution}}work attribution: {{attribution}}\n{{/if}}",
    "provision": "{{#if provision}}provision information: {{provision}}\n{{/if}}",
}

TEMPLATES["composite"] = (
    f'{TEMPLATES["title"]}{TEMPLATES["person"]}{TEMPLATES["roles"]}{TEMPLATES["attribution"]}{TEMPLATES["provision"]}'
)


def eprintln(string):
    print(string, file=sys.stderr)


def template_records():
    ctx = build_session_context()
    dataframe = ctx.table("csv")

    eprintln("templating...")
    tpl.write_templated_fields(
        dataframe,
        TEMPLATES,
        "output/templated/",
        id_column="id",
        columns_of_interest=[
            "title",
            "person",
            "roles",
            "attribution",
            "provision",
        ],
    )


def dedup_records():
    ctx = build_session_context()

    eprintln("dedupping...")
    dedup.dedup_from_into(ctx, f"output/templated/", "output/dedup/")


def vectorize_records():
    ctx = build_session_context()

    eprintln("vectorizing...")
    embed.vectorize(
        ctx, "output/dedup/", "output/vectors/", model=MODEL, dimension=EMBEDDING_SIZE
    )


def field_vectors(
    ctx: df.SessionContext,
    key: str,
    configuration: Optional[Dict] = None,
) -> torch.Tensor:
    if configuration is None:
        # defaults to OpenAI dimensions / datatype
        configuration = {"dimensions": EMBEDDING_SIZE, "field_type": "float32"}
    result = (
        ctx.table("templated_vectors")
        .filter(df.col("key") == key)
        .select(df.col("embedding"))
    )
    size = result.count()
    dim = configuration["dimensions"]
    field_type = configuration["field_type"]
    dtype = name_to_torch_type(field_type)
    tensor = torch.empty((size, dim), dtype=dtype, device="cuda")
    dataframe_to_tensor(result, tensor)
    return tensor


def write_field_averages(
    ctx: df.SessionContext,
    key: str,
    destination: str,
):
    fv = field_vectors(ctx, key)
    average = torch.mean(fv, 0).cpu().detach().numpy()
    pandas_df = pd.DataFrame({"template": [key], "average": [average]})
    datafusion_df = ctx.from_arrow(pandas_df)
    datafusion_df.write_parquet(destination)


def average_fields():
    ctx = build_session_context()

    eprintln("averaging (for imputation)...")
    for key in TEMPLATES.keys():
        write_field_averages(ctx, key, "output/vector_averages/")


def build_index_map():
    ctx = build_session_context()

    ctx.table("templated").filter(df.col("key") == "composite").sort(
        df.col("hash")
    ).select(
        (df.functions.row_number() - 1).alias("vector_id"),
        df.col("id"),
        df.col("hash"),
    ).write_parquet(
        "output/index_map/"
    )


def load_vectors(ctx: df.SessionContext) -> torch.Tensor:
    embeddings = (
        ctx.table("index_vectors").sort(df.col("vector_id")).select(df.col("embedding"))
    )
    count = embeddings.count()

    vectors = torch.empty((count, EMBEDDING_SIZE), dtype=torch.float32, device="cuda")
    dataframe_to_tensor(embeddings, vectors)

    return vectors


def index_field():
    ctx = build_session_context()
    vectors = load_vectors(ctx)

    ann = ANN(vectors, beam_size=32)
    print(ann.dump_logs())

    distances = tensor_to_arrow(ann.distances)
    beams = tensor_to_arrow(ann.beams)
    table = pa.Table.from_pydict({"beams": beams, "distances": distances})
    # todo this should not be plural names
    ctx.from_arrow_table(table).select(
        (df.functions.row_number() - 1).alias("vector_id"),
        df.col("beams"),
        df.col("distances"),
    ).write_parquet("output/ann/")
    return ann


def discover_training_set():
    """
    1. Loads the existing ANN
    2. Finds the first peak of the first derivative
    3. Keeping a tally to balance, guesses on the left or right of the first peak threshold
    4. Sends records (in toto) to LLM
    4. Writes each match or non-match in a training table
    """

    # 1.
    print("Loading ann...")
    ann = load_ann()
    distances = ann.distances
    (vector_count, beam_size) = distances.size()

    # 2.
    print("Finding peaks...")
    sample_size = 1000
    sample_size = min(vector_count, sample_size)
    (all_distances, _) = distances[0:sample_size].flatten().sort()
    (length,) = all_distances.size()
    tail = all_distances[1:length]
    head = all_distances[0 : length - 1]
    diff = tail - head
    # maybe use smoothing (savitzky_golay?) for smoothing first to remove jitter?
    (peaks, _) = scipy.signal.find_peaks(diff.cpu().numpy())
    # Assume first peak is good for now
    if len(peaks) < 1:
        raise Exception("Unable to find peaks in derivative of distances")
    first_peak = peaks[0]
    threshold = all_distances[first_peak]

    # 3.
    print("Asking oracle...")
    ctx = build_session_context()
    candidate_size = 1000  # increase for better training
    same = 0
    different = 0
    record = []
    candidates = []
    for i in range(0, candidate_size):
        print(f"evaluating record {i}/{candidate_size}")
        beam = ann.beams[i]
        distance = ann.distances[i]
        indices = (distance > threshold).nonzero()
        (count, _) = indices.size()
        if len(indices) == 0:
            continue
        pivot = indices[0][0]
        total = same + different
        if same > total / 2 and pivot < len(distance):
            j = beam[pivot]
        else:
            j = beam[0]

        (answer, id1, id2) = ask_oracle_with_vid(ctx, int(i), int(j))
        if answer == True:
            same += 1
        else:
            different += 1
        # This really should be rename to `left_tid` / `right_tid`
        record = {"match": answer, "left": id1, "right": id2}
        candidates.append(record)

    print("Writing results")
    candidates_pd = pd.DataFrame(candidates)
    ctx.from_pandas(candidates_pd).write_parquet("output/training_set/")


def get_record_from_vid(ctx, vid) -> Dict:
    templated_df = (
        ctx.table("templated")
        .filter(df.col("key") == "composite")
        .select(df.col("templated"), df.col("id").alias("tid"))
    )
    result = (
        ctx.table("index_map")
        .filter(vid == df.col("vector_id"))
        .join(templated_df, left_on="id", right_on="tid", how="inner")
        .select(df.col("id"), df.col("templated"))
        .limit(1)
    )
    return result.to_pylist()[0]


def check_y_or_n(string):
    if re.search(r".*[Y|y]es\W*$", string, re.MULTILINE) is None:
        return False
    else:
        return True


def ask_oracle_with_vid(ctx, vid1, vid2):
    """
    1. Map from vid to record id
    2. load record 1 and 2
    3. Ask LLM for match of record 1 and 2
    """
    record1 = get_record_from_vid(ctx, vid1)
    record2 = get_record_from_vid(ctx, vid2)

    y_or_n = ask_oracle(record1["templated"], record2["templated"])

    return (y_or_n, record1["id"], record2["id"])


def ask_oracle_with_id(ctx, id1, id2):
    record1 = ctx.sql(
        f"SELECT templated FROM templated WHERE key = 'composite' AND id = {id1}"
    ).to_pylist()[0]
    record2 = ctx.sql(
        f"SELECT templated FROM templated WHERE key = 'composite' AND id = {id2}"
    ).to_pylist()[0]
    return ask_oracle(record1["templated"], record2["templated"])


def ask_oracle(s1, s2):
    subject = "Historic People who are creators of works"
    client = OpenAI()

    prompt = "You are a classifier deciding if two people are the same or not not."
    question = f"""Tell me whether the following two records are referring to the same person or a different person using a chain of reasoning followed by a single yes or no answer on a single line, without any formatting.

1:  {s1}

2:  {s2}
"""
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            },
        ],
    )
    print(f">> {question}")
    content = completion.choices[0].message.content
    print(f"\n<< {content}")
    verdict = check_y_or_n(completion.choices[0].message.content)
    print(f"verdict: {verdict}")

    return verdict


def load_ann() -> ANN:
    ctx = build_session_context()

    print("loading ann..")
    ann = ANN.load_from_dataframe(ctx.table("total_ann"))

    print(ann.dump_logs())
    return ann


def candidate_field_distances(ctx, candidates: df.DataFrame, destination: str):
    averages = ctx.table("vector_averages").to_pandas()
    vectors = ctx.table("vectors")
    for key in TEMPLATES.keys():
        print(f"processing key {key}")
        average_for_key = pa.array(
            averages[averages["template"] == key].iloc[0, 1]
        ).to_numpy()
        field = (
            ctx.table("templated")
            .filter(df.col("key") == key)
            .select(df.col("id"), df.col("hash"), df.col("templated").alias(key))
        )
        left_match = candidates.join(
            field.select(df.col("id"), df.col("hash").alias("left_hash")),
            how="left",
            left_on="left",
            right_on="id",
        ).drop("id")
        field_match = left_match.join(
            field.select(df.col("id"), df.col("hash").alias("right_hash")),
            how="left",
            left_on="right",
            right_on="id",
        )

        vector_comparisons = (
            field_match.join(
                vectors.with_column_renamed("embedding", f"left_embedding"),
                how="left",
                left_on="left_hash",
                right_on="hash",
            )
            .drop("hash")
            .join(
                vectors.with_column_renamed("embedding", "right_embedding"),
                how="left",
                left_on="right_hash",
                right_on="hash",
            )
            .select(
                df.functions.coalesce(
                    df.col("left_embedding"),
                    df.lit(average_for_key).cast(
                        pa.list_(pa.float32(), EMBEDDING_SIZE)
                    ),
                ).alias(f"left_embedding"),
                df.functions.coalesce(
                    df.col("right_embedding"),
                    df.lit(average_for_key).cast(
                        pa.list_(pa.float32(), EMBEDDING_SIZE)
                    ),
                ).alias(f"right_embedding"),
                df.col("left").alias("left_id"),
                df.col("right").alias("right_id"),
            )
        ).sort(df.col("left_id"), df.col("right_id"))
        size = vector_comparisons.count()
        print(f"total comparisons: {size}")
        left_tensor = torch.empty(
            (size, EMBEDDING_SIZE), dtype=torch.float32, device="cuda"
        )
        dataframe_to_tensor(
            vector_comparisons.select(df.col("left_embedding")), left_tensor
        )
        right_tensor = torch.empty(
            (size, EMBEDDING_SIZE), dtype=torch.float32, device="cuda"
        )
        dataframe_to_tensor(
            vector_comparisons.select(df.col("right_embedding")), right_tensor
        )
        distance = torch.clamp(
            (1 - (left_tensor * right_tensor).sum(dim=1)) / 2, min=0.0, max=1.0
        )
        distance_arrow = tensor_to_arrow(distance)
        distance_table = pa.table(
            vector_comparisons.select(
                df.col("left_id"), df.col("right_id"), df.lit(key).alias("key")
            )
        ).append_column("distance", distance_arrow)

        ctx.from_arrow(distance_table).write_parquet(destination)


def calculate_training_field_distances():
    print("Calculating training field distances...")
    ctx = build_session_context()
    candidates = ctx.table("training_set")
    candidate_field_distances(ctx, candidates, "output/match_field_distances/")


def train_weights(destination="output/weights"):
    """
    b := weights in order, 1D tensor
    x := [1.0, distance keys in order], 2D tensor, batch sorted by left,right
    y = sigma( x . b )
    y_hat := match value, 1D tensor
    """
    ctx = build_session_context()
    keys = sorted(list(TEMPLATES.keys()))

    field_distances = ctx.table("match_field_distances")
    x = get_field_distances(ctx, field_distances)
    (batch_size, _) = x.size()
    x = x.cpu().detach().numpy()

    matches = (
        ctx.table("training_set")
        .sort(df.col("left"), df.col("right"))
        .select(df.col("match").cast(pa.float32()).alias("y"))
    )
    y = matches.to_pandas()["y"].to_numpy()

    training_size = int(batch_size * 2 / 3)

    (x, y) = shuffle(x, y, random_state=23)
    x_train = x[0:training_size]
    x_test = x[training_size:]

    y_train = y[0:training_size]
    y_test = y[training_size:]

    logr = linear_model.LogisticRegression(verbose=5, solver="liblinear")
    logr.fit(x_train, y_train)

    y_predicted = logr.predict(x_test)
    auc = metrics.roc_auc_score(y_test, y_predicted)
    print(f"\nROC AUC: {auc}")

    coef = logr.coef_[0].astype(np.float32())
    intercept = logr.intercept_.astype(np.float32())
    weights = np.concatenate([intercept, coef])
    weight_object = dict(zip(["intercept"] + keys, map(lambda w: [w], weights)))
    ctx.from_pydict(weight_object).write_parquet(destination)


def show_weights() -> df.DataFrame:
    ctx = build_session_context()
    return ctx.table("weights")


def classify(x, weights):
    (batch_size, _) = x.size()
    x = torch.concat(
        [torch.ones(batch_size, dtype=torch.float32, device="cuda").unsqueeze(1), x],
        dim=1,
    )
    y_hat = torch.special.expit((x * weights).sum(dim=1))
    return y_hat


def search_string(query: str, ann: Optional[ANN] = None) -> df.DataFrame:
    if ann is None:
        ann = load_ann()
    ctx = build_session_context()
    result = (
        ann_search(ctx, ann, query)
        .sort(df.col("distance"))
        .select(
            df.col("distance"),
            df.col("attribution"),
            df.col("person"),
            df.col("provision"),
            df.col("roles"),
            df.col("title"),
        )
    )

    return result


def search():
    parser = argparse.ArgumentParser(usage="search [query] [options]")
    parser.add_argument("query", help="The query to search for")
    args = parser.parse_args()
    result = search_string(args.query)

    return result


def ann_search(ctx: df.SessionContext, ann: ANN, query_string: str) -> df.DataFrame:
    client = OpenAI()

    response = client.embeddings.create(input=query_string, model=MODEL)
    embedding = response.data[0].embedding
    query_tensor = torch.tensor(embedding, dtype=torch.float32, device="cuda").reshape(
        (1, EMBEDDING_SIZE)
    )

    result = ann.search(query_tensor)
    matches = pa.array(result.indices.flatten().cpu().numpy())
    distances = pa.array(result.distances.flatten().cpu().numpy())
    results = ctx.from_arrow(
        pa.RecordBatch.from_arrays([matches, distances], ["match", "distance"])
    )

    records = ctx.table("csv")
    index_map = ctx.table("index_map")

    return (
        results.join(index_map, left_on="match", right_on="vector_id")
        .with_column_renamed("id", "match_id")
        .join(records, left_on="match_id", right_on="id")
    )


def filter_candidates():
    print("filtering for candidate matches...")
    ctx = build_session_context()
    ann = load_ann()
    (vector_count, beam_length) = ann.beams.size()
    threshold = 0.3  # Influence the number of checked pairs
    index_map = pa.table(ctx.table("index_map")).to_pandas()
    left = []
    right = []
    for i in range(0, vector_count):
        if i % 1000 == 0 and left is not []:
            print(f"processing {i}th record")
            results = {"left": left, "right": right}
            ctx.from_pydict(results).write_parquet("output/filtered/")
            left = []
            right = []
        tid1 = index_map[index_map["vector_id"] == i]["id"].to_numpy()
        distances = ann.distances[i]
        positions = distances < threshold
        indices = ann.beams[i][positions].cpu().detach().numpy()
        tids = index_map[index_map["vector_id"].isin(indices)]["id"].to_numpy()
        left += list(tid1.repeat(len(tids)))
        right += list(tids)
    if left is not []:
        results = {"left": left, "right": right}
        ctx.from_pydict(results).write_parquet("output/filtered/")


def calculate_field_distances():
    print("Calculating filter candidate field distances...")
    ctx = build_session_context()
    candidates = ctx.table("filtered")
    candidate_field_distances(ctx, candidates, "output/field_distances/")


def get_field_distances(ctx, source: df.DataFrame) -> torch.Tensor:
    field_distances = (
        source.aggregate(
            [df.col("left_id"), df.col("right_id")],
            [
                df.functions.array_agg(
                    df.col("distance"), order_by=[df.col("key")]
                ).alias("distances")
            ],
        )
        .sort(df.col("left_id"), df.col("right_id"))
        .select(df.col("distances"))
    )
    size = field_distances.count()
    x = torch.empty((size, len(TEMPLATES)), dtype=torch.float32, device="cuda")
    dataframe_to_tensor(field_distances, x)
    return x


def classify_record_matches():
    print("Classifying record matches...")
    ctx = build_session_context()
    vectors = load_vectors(ctx)

    weights = torch.tensor(
        ctx.table("weights").to_pandas().to_numpy()[0],
        dtype=torch.float32,
        device="cuda",
    )

    field_distances = ctx.table("field_distances")
    (weight_count,) = weights.size()
    key_count = weight_count - 1  # minus intercept
    x = get_field_distances(ctx, field_distances)
    print("beginning classification...")
    y_hat = classify(x, weights)
    filtered = ctx.table("filtered").sort(df.col("left"), df.col("right")).to_pandas()
    filtered["prediction"] = y_hat.cpu().detach().numpy()
    print("writing predictions...")
    ctx.from_pandas(filtered).write_parquet("output/prediction")


def build_clusters():
    inclusion_threshold = 0.97  # 0.8  # 0.86
    ctx = build_session_context()
    ids = ctx.table("csv").select(df.col("id")).to_pydict()["id"]
    disjoint_set = scipy.cluster.hierarchy.DisjointSet(ids)
    matches = ctx.table("prediction").filter(df.col("prediction") > inclusion_threshold)
    for batch in matches.execute_stream():
        batch = batch.to_pyarrow().to_pandas()
        for _, record in batch.iterrows():
            left = record["left"]
            right = record["right"]
            disjoint_set.merge(left, right)

    x = disjoint_set.subsets()
    cluster_ids = []
    cluster_elements = []
    for cluster_id, cluster in enumerate(x):
        for cluster_element in cluster:
            cluster_ids.append(cluster_id)
            cluster_elements.append(cluster_element)
    ctx.from_pydict(
        {"cluster_id": cluster_ids, "cluster_element": cluster_elements}
    ).write_parquet("output/clusters/")


def embedding_for_id(ctx: df.SessionContext, key: str, tid: int) -> np.ndarray:
    averages = ctx.table("vector_averages").to_pandas()
    vectors = ctx.table("vectors")
    average_for_key = pa.array(
        averages[averages["template"] == key].iloc[0, 1]
    ).to_numpy()

    embedding = (
        ctx.table("templated")
        .filter(df.col("id") == tid)
        .filter(df.col("key") == key)
        .limit(1)
        .select(df.col("id"), df.col("hash").alias("field_hash"))
        .join(vectors, how="inner", left_on="field_hash", right_on="hash")
        .select(df.col("embedding"))
        .to_pandas()["embedding"]
        .to_numpy()
    )

    if len(embedding) == 0:
        print(f"using average for {key} on {tid}")
        return average_for_key
    else:
        return embedding[0]


def calculate_record_distance(left: int, right: int) -> float:
    ctx = build_session_context()
    distances = [1.0]
    vectors = ctx.table("vectors")
    keys = sorted(list(TEMPLATES.keys()))
    for key in keys:
        left_embedding = embedding_for_id(ctx, key, left)
        right_embedding = embedding_for_id(ctx, key, right)

        distance = (1 - (left_embedding * right_embedding).sum()) / 2
        distances.append(distance)

    weights = ctx.table("weights").to_pandas().to_numpy()[0]

    return scipy.special.expit((distances * weights).sum())


def calculate_expanded_match(frame: df.DataFrame) -> df.DataFrame:
    right_frame = frame.with_column_renamed(
        "cluster_id", "right_cluster_id"
    ).with_column_renamed("cluster_element", "right_cluster_element")

    return (
        frame.join(
            right_frame, how="inner", left_on="cluster_id", right_on="right_cluster_id"
        )
        .filter(df.col("cluster_element") < df.col("right_cluster_element"))
        .select(
            df.col("cluster_element").alias("left"),
            df.col("right_cluster_element").alias("right"),
        )
        .sort(df.col("left"), df.col("right"))
    )


def calculate_adjusted_rand_score():
    ctx = build_session_context()
    orig = ctx.table("csv").sort(df.col("id")).select('"CID"').to_pydict()["CID"]

    ours = (
        ctx.table("clusters")
        .sort(df.col("cluster_element"))
        .select("cluster_id")
        .to_pydict()["cluster_id"]
    )

    score = adjusted_rand_score(orig, ours)

    return score


def recall():
    ctx = build_session_context()

    records = ctx.table("matches")
    ## This needs to take matches from the data matches csv
    original_match = (
        records.select(
            df.col('"Source"').alias("left"), df.col('"Target"').alias("right")
        )
        .sort(df.col("left"), df.col("right"))
        .select(
            df.functions.array(df.col("left"), df.col("right")).alias("original_pair")
        )
    )

    clusters = ctx.table("clusters").select(
        df.col("cluster_id"), df.col("cluster_element")
    )
    our_match = (
        clusters.select(
            df.col("cluster_element").alias("left"),
            df.col("cluster_id").alias("cid_left"),
        )
        .join(
            clusters.select(
                df.col("cluster_element").alias("right"),
                df.col("cluster_id").alias("cid_right"),
            ),
            how="inner",
            left_on="cid_left",
            right_on="cid_right",
        )
        .select(df.col("left"), df.col("right"))
        .sort(df.col("left"), df.col("right"))
        .filter(df.col("left") < df.col("right"))
        .select(
            df.functions.array(df.col("left"), df.col("right")).alias("predicted_pair")
        )
    )

    true_positives = our_match.join(
        original_match, how="inner", left_on="predicted_pair", right_on="original_pair"
    ).count()
    false_positives = our_match.join(
        original_match, how="anti", left_on="predicted_pair", right_on="original_pair"
    )
    print("False positives...")
    print(false_positives)
    false_positives = false_positives.count()
    false_negatives = original_match.join(
        our_match, how="anti", left_on="original_pair", right_on="predicted_pair"
    ).count()
    print(f"true positives: {true_positives}")
    print(f"false positives: {false_positives}")
    print(f"false negatives: {false_negatives}")

    precision = true_positives / (true_positives + false_positives)
    fdr = false_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * true_positives / (2 * true_positives + false_positives + false_negatives)

    print(f"precision: {precision}")
    print(f"false discovery rate: {fdr}")
    print(f"recall: {recall}")
    print(f"F1: {f1}")


def openai_compare_strings(
    s1: str,
    s2: str,
    model: Literal[
        "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
    ] = "text-embedding-3-small",
) -> float:
    e1 = np.array(oa.embeddings.create(input=s1, model=model).data[0].embedding)
    print(len(e1))
    e2 = np.array(oa.embeddings.create(input=s2, model=model).data[0].embedding)
    return float((1 - (e1 * e2).sum()) / 2)


def cosine_distance(v1, v2):
    return float((1 - (v1 * v2).sum()) / 2)


def openai_embedding(
    s: str,
    model: Literal[
        "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
    ] = "text-embedding-3-small",
) -> np.ndarray:
    return np.array(oa.embeddings.create(input=s, model=model).data[0].embedding)


def openai_compare_fields(
    ctx: df.SessionContext,
    tid1: int,
    tid2: int,
    key: str,
    model="text-embedding-3-small",
) -> float:
    s1 = ctx.sql(
        f"""select templated from templated where id = {tid1} and key = '{key}'"""
    ).to_pylist()[0]["templated"]
    s2 = ctx.sql(
        f"""select templated from templated where id = {tid2} and key = '{key}'"""
    ).to_pylist()[0]["templated"]
    return openai_compare_strings(s1, s2, model)


def openai_compare_records_templated(
    ctx: df.SessionContext,
    tid1: str,
    tid2: str,
    template: str,
    model="text-embedding-3-small",
) -> float:
    compiler = pybars.Compiler()
    compiled_template = compiler.compile(template)
    r1 = ctx.sql(f"""select * from csv where id = {tid1}""").to_pylist()[0]
    r2 = ctx.sql(f"""select * where id = {tid2}""").to_pylist()[0]

    s1 = compiled_template(r1)
    s2 = compiled_template(r2)

    return openai_compare_strings(s1, s2, model)


def main():
    template_records()
    dedup_records()
    vectorize_records()
    average_fields()
    build_index_map()
    index_field()
    discover_training_set()
    calculate_training_field_distances()
    train_weights()
    filter_candidates()
    calculate_field_distances()
    classify_record_matches()
    build_clusters()


if __name__ == "__main__":
    main()
