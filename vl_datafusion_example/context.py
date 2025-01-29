import datafusion as df
import pyarrow as pa

RECORD_SCHEMA = pa.schema(
    [
        pa.field("id", pa.int64(), nullable=False),
        pa.field("field1", pa.string(), nullable=True),
        pa.field("field2", pa.string(), nullable=True),
        pa.field("field3", pa.string(), nullable=True),
    ]
)


def build_session_context(location="output/") -> df.SessionContext:
    ctx = df.SessionContext()
    ctx.register_parquet("records", f"{location}records/", schema=RECORD_SCHEMA)
    # register more here

    return ctx
