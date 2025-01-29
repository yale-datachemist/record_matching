from .context import build_session_context


def main():
    ctx = build_session_context()

    ctx.sql("""select 'hello, world!' as welcome_text""").show()


if __name__ == "__main__":
    main()
