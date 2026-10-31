"""Agent Forge Lab CLI; offline unless the user explicitly selects provider."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .engine import MultiAgentEngine
from .knowledge import KnowledgeBase
from .providers import OpenAIResponsesModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only, evidence-grounded multi-agent research")
    parser.add_argument("command", choices=["run"])
    parser.add_argument("knowledge_base", type=Path, help="Synthetic JSON document manifest")
    parser.add_argument("--question", required=True, help="Research question")
    parser.add_argument("--output", type=Path, help="Write JSON report")
    parser.add_argument("--provider", choices=["offline", "openai"], default="offline")
    parser.add_argument("--allow-external-data-transfer", action="store_true")
    parser.add_argument("--model", help="OpenAI model identifier (or AGENT_FORGE_MODEL)")
    parser.add_argument("--max-tool-calls", type=int, default=8)
    args = parser.parse_args()
    if args.provider == "offline" and (args.allow_external_data_transfer or args.model):
        parser.error("offline mode does not accept API consent or model")
    model = (OpenAIResponsesModel(model=args.model,
                                 allow_external_data_transfer=args.allow_external_data_transfer)
             if args.provider == "openai" else None)
    report = MultiAgentEngine(KnowledgeBase.from_json(args.knowledge_base), model=model,
                              max_tool_calls=args.max_tool_calls).run(args.question)
    result = json.dumps(report.to_dict(), indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
        print(f"Wrote {len(report.claims)} verified-quote claims to {args.output}")
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
