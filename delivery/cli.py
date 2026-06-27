"""CLI 入口：ingest / ask / stats / history。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cmd_ingest(args):
    from pipeline.runtime import get_indexing_pipeline
    p = get_indexing_pipeline()
    for f in args.files:
        r = p.ingest(f)
        print(f"[{r.status}] {f}: {r.detail}")


def cmd_ask(args):
    from pipeline.runtime import get_generation_pipeline
    from memory.session import new_session, load_session, get_session
    if args.new_session:
        new_session()
    elif args.session_id:
        load_session(args.session_id)
    g = get_generation_pipeline()
    answer, sources = g.retrieve_and_format(args.question, args.session_id)
    print(answer)
    if sources:
        print("\n来源:")
        for s in sources:
            print(f"  - {s['file']} 第{s['page']}页")


def cmd_stats(_args):
    from pipeline.runtime import get_clip_index, get_bge_index
    print(f"CLIP (image): {get_clip_index().count}")
    print(f"BGE  (text) : {get_bge_index().count}")


def cmd_history(args):
    from memory.session import load_session, list_sessions
    if args.session_id:
        s = load_session(args.session_id)
        for m in s.messages:
            tag = "Q" if m.get("role") == "user" else "A"
            print(f"[{tag}] {m.get('content', '')[:200]}")
    else:
        for s in list_sessions():
            print(f"{s['session_id'][:8]}...  {s['created_at']}  {s['message_count']} msgs")


def main():
    p = argparse.ArgumentParser(description="多模态 RAG (LlamaIndex)")
    sub = p.add_subparsers(dest="command")

    pi = sub.add_parser("ingest"); pi.add_argument("files", nargs="+"); pi.set_defaults(func=cmd_ingest)
    pa = sub.add_parser("ask"); pa.add_argument("question"); pa.add_argument("-n", "--new-session", action="store_true"); pa.add_argument("-s", "--session-id"); pa.set_defaults(func=cmd_ask)
    ps = sub.add_parser("stats"); ps.set_defaults(func=cmd_stats)
    ph = sub.add_parser("history"); ph.add_argument("-s", "--session-id"); ph.set_defaults(func=cmd_history)

    args = p.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
