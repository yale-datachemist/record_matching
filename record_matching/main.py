from .context import build_session_context, EMBEDDING_SIZE, MODEL
from vectorlink_py import template as tpl, dedup, embed
import sys


'''
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
'''

TEMPLATES = {
    "title": "{{#if title}}work title: {{title}}\n{{/if}}",
    "person": "{{#if person}}person name: {{person}}\n{{/if}}",
    "roles": "{{#if roles}}person roles: {{roles}}\n{{/if}}",
    "attribution": "{{#if attribution}}work attribution: {{attribution}}\n{{/if}}",
    "provision": "{{#if provision}}provision information: {{provision}}\n{{/if}}",
}

TEMPLATES["composite"] = f'{TEMPLATES["title"]}{TEMPLATES["person"]}{TEMPLATES["roles"]}{TEMPLATES["attribution"]}{TEMPLATES["provision"]}'

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
        id_column='id',
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


def main():
    template_records()
    dedup_records()
    vectorize_records()


if __name__ == "__main__":
    main()
