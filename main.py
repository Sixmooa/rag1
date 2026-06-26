"""多模态 RAG 系统统一入口。"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import settings
from db.chroma_store import get_store
from rag.qa_engine import get_qa_engine
from memory.session import new_session, get_session, load_session, list_sessions
from tools.file_parser import detect_file_type


def cmd_ingest(args):
    """入库文件（支持 PDF / 图片 / TXT / MD）。"""
    store = get_store()
    for fpath in args.files:
        file_type = detect_file_type(fpath)
        if file_type == 'pdf':
            store.add_pdf(fpath)
        elif file_type == 'image':
            store.add_image(fpath)
        elif file_type in ('txt', 'md'):
            store.add_text_file(fpath)
        else:
            print(f"不支持的文件类型: {file_type} ({fpath})")


def cmd_ask(args):
    """RAG 问答。"""
    if args.new_session:
        new_session()
        print(f"已创建新会话: {get_session().session_id}")
    elif args.session_id:
        load_session(args.session_id)
        print(f"已加载会话: {args.session_id}")

    engine = get_qa_engine()
    engine.ask(args.question, top_k=args.top_k)


def cmd_stats(_args):
    """显示数据库统计。"""
    store = get_store()
    print(f"CLIP collection (图像向量): {store.clip_count} 条记录")
    print(f"BGE  collection (文本向量): {store.text_count} 条记录")


def cmd_history(args):
    """查看会话历史。"""
    if args.session_id:
        sess = load_session(args.session_id)
        print(f"会话: {sess.session_id}")
        print(f"创建时间: {sess.data.get('created_at')}")
        print("-" * 60)
        for msg in sess.messages:
            tag = "Q" if msg.get("role") == "user" else "A"
            print(f"[{tag}] {msg.get('timestamp', '')}  {msg['content'][:200]}")
            if msg.get("sources"):
                for s in msg["sources"]:
                    print(f"     -> {s['file']} 第{s['page']}页 (RRF: {s['rrf_score']})")
    else:
        sessions = list_sessions()
        if not sessions:
            print("暂无会话记录。")
            return
        print(f"{'会话ID':<38} {'创建时间':<20} {'消息数':>6}")
        print("-" * 68)
        for s in sessions:
            print(f"{s['session_id']:<38} {s['created_at']:<20} {s['message_count']:>6}")


def main():
    parser = argparse.ArgumentParser(description="多模态 RAG 系统")
    sub = parser.add_subparsers(dest="command")

    # ingest
    p_ingest = sub.add_parser("ingest", help="入库文件")
    p_ingest.add_argument("files", nargs="+", help="文件路径（PDF/图片/TXT/MD）")
    p_ingest.set_defaults(func=cmd_ingest)

    # ask
    p_ask = sub.add_parser("ask", help="RAG 问答")
    p_ask.add_argument("question", help="问题")
    p_ask.add_argument("-k", "--top-k", type=int, default=None, help="检索结果数")
    p_ask.add_argument("-n", "--new-session", action="store_true", help="创建新会话")
    p_ask.add_argument("-s", "--session-id", type=str, default=None, help="加载已有会话ID")
    p_ask.set_defaults(func=cmd_ask)

    # stats
    p_stats = sub.add_parser("stats", help="数据库统计")
    p_stats.set_defaults(func=cmd_stats)

    # history
    p_hist = sub.add_parser("history", help="查看会话历史")
    p_hist.add_argument("-s", "--session-id", type=str, default=None,
                        help="指定会话ID查看详情，不指定则列出所有会话")
    p_hist.set_defaults(func=cmd_history)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
