import datafusion as df
import pyarrow as pa

EMBEDDING_SIZE=3072
MODEL="text-embedding-3-large"

RECORD_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("field1", pa.string(), nullable=True),
        pa.field("field2", pa.string(), nullable=True),
        pa.field("field3", pa.string(), nullable=True),
    ]
)

TEMPLATED_SCHEMA = pa.schema(
    [
        pa.field("id", pa.string_view(), nullable=False),
        pa.field("templated", pa.string_view(), nullable=False),
        pa.field("hash", pa.string_view(), nullable=False),
    ]
)

DEDUP_SCHEMA = pa.schema(
    [
        pa.field("hash", pa.string_view(), nullable=False),
        pa.field("templated", pa.string_view(), nullable=False),
    ]
)


def build_session_context(input="yale_people_v3/",location="output/") -> df.SessionContext:
    ctx = df.SessionContext()
    ctx.register_csv("csv", f"{input}benchmark_data_records.csv")
    #ctx.register_parquet("records", f"{location}records/", schema=RECORD_SCHEMA)

    ctx.register_parquet(
        f"templated",
        f"{location}templated/",
        schema=TEMPLATED_SCHEMA,
        table_partition_cols=[("key", "string")],
    )
    ctx.register_parquet("dedup", f"{location}dedup/", schema=DEDUP_SCHEMA)


    # register more here

    return ctx
