import json
import unittest
from pathlib import Path

from agent_forge_lab.audit import AuditTrail, digest
from agent_forge_lab.engine import (
    MultiAgentEngine, ReadOnlyTools, ToolBudgetError, Candidate, INSTRUCTIONS,
)
from agent_forge_lab.knowledge import Document, KnowledgeBase

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "synthetic_kb.json"
QUESTION = "Summarize login investigations and approved maintenance"


class StubModel:
    def __init__(self, proposals):
        self.proposals = proposals
        self.calls = []

    def ask(self, role, instructions, context):
        self.calls.append((role, instructions, context))
        response = self.proposals[role]
        return response(context) if callable(response) else response


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase.from_json(SAMPLE)

    def test_four_offline_roles_and_grounded_report(self):
        report = MultiAgentEngine(self.kb).run(QUESTION)
        self.assertEqual(report.mode, "offline")
        self.assertEqual(report.status, "needs_human_review")
        self.assertEqual(report.metrics["model_calls"], 0)
        self.assertEqual(len(report.agent_messages), 4)
        self.assertEqual([m["role"] for m in report.agent_messages],
                         ["planner", "researcher", "verifier", "reporter"])
        self.assertGreaterEqual(len(report.claims), 1)
        self.assertTrue(all(c.quote in self.kb.read(c.doc_id).text for c in report.claims))
        self.assertTrue(all("[" + c.doc_id + "]" in report.answer for c in report.claims))
        self.assertEqual(report.metrics["audit_valid"], True)

    def test_engine_is_deterministic_offline(self):
        a = MultiAgentEngine(self.kb).run(QUESTION)
        b = MultiAgentEngine(self.kb).run(QUESTION)
        self.assertEqual(a.to_dict(), b.to_dict())

    def test_read_only_tools_refuse_remediation(self):
        trace = AuditTrail()
        tools = ReadOnlyTools(self.kb, trace)
        for name in ("disable_user", "delete_files", "run_shell", "send_email"):
            with self.subTest(name=name):
                with self.assertRaises(PermissionError):
                    tools.invoke(name, account="admin")
        self.assertEqual(tools.calls, 0)

    def test_tool_budget_enforced(self):
        tools = ReadOnlyTools(self.kb, AuditTrail(), max_calls=1)
        tools.invoke("search_documents", query="login")
        with self.assertRaises(ToolBudgetError):
            tools.invoke("search_documents", query="login")

    def test_unknown_question_no_hallucinated_answer(self):
        report = MultiAgentEngine(self.kb).run("xylophonist xylophonic")
        self.assertEqual(report.claims, ())
        self.assertIn("No source-verified", report.answer)
        self.assertEqual(report.metrics["model_calls"], 0)

    def test_invalid_questions(self):
        for q in ["", " ", "x" * 501, 1]:
            with self.subTest(q=str(q)[:12]):
                with self.assertRaises(ValueError):
                    MultiAgentEngine(self.kb).run(q)

    def test_invalid_engine_configurations(self):
        for values in [{"max_claims": 0}, {"max_claims": True},
                       {"max_tool_calls": 1}, {"max_tool_calls": True}]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    MultiAgentEngine(self.kb, **values)

    def test_document_injection_is_never_executed(self):
        report = MultiAgentEngine(self.kb).run("untrusted injection instructions example")
        self.assertEqual(report.status, "needs_human_review")
        self.assertNotIn("disable_user", [e["action"] for e in report.audit])
        self.assertEqual(report.metrics["model_calls"], 0)
        self.assertTrue(report.metrics["audit_valid"])

    def test_audit_does_not_store_raw_documents(self):
        report = MultiAgentEngine(self.kb).run(QUESTION)
        audit = json.dumps(report.audit)
        self.assertNotIn("Maintenance change CHG-SYN-202 was approved", audit)
        self.assertNotIn("Ignore all previous instructions", audit)

    def test_agent_to_tool_calls_are_bounded(self):
        report = MultiAgentEngine(self.kb, max_tool_calls=2).run(QUESTION)
        self.assertEqual(report.metrics["read_only_tool_calls"], 2)
        self.assertLessEqual(report.metrics["retrieved_documents"], 1)

    def test_planner_cannot_request_any_tool(self):
        model = StubModel({"planner": {"queries": [QUESTION], "tool": "delete"}})
        with self.assertRaisesRegex(ValueError, "planner output schema"):
            MultiAgentEngine(self.kb, model=model).run(QUESTION)
        self.assertEqual(len(model.calls), 1)

    def test_planner_rejects_invalid_queries(self):
        for query_list in ([], [""], ["a"] * 4, ["a" * 241]):
            with self.subTest(queries=query_list[:1]):
                model = StubModel({"planner": {"queries": query_list}})
                with self.assertRaises(ValueError):
                    MultiAgentEngine(self.kb, model=model).run(QUESTION)

    def test_four_separate_model_roles_can_select_grounded_claims(self):
        quote = "The recorded successful administrator login occurred during that maintenance window."
        claim_id = "CLM-" + digest(["INC-102", quote])[:12].upper()
        model = StubModel({
            "planner": {"queries": ["approved maintenance login"]},
            "researcher": {"candidates": [{"document_id": "INC-102", "quote": quote}]},
            "verifier": {"accepted_claim_ids": [claim_id]},
            "reporter": {"selected_claim_ids": [claim_id]},
        })
        report = MultiAgentEngine(self.kb, model=model).run(QUESTION)
        self.assertEqual(report.mode, "model-assisted")
        self.assertEqual(report.metrics["model_calls"], 4)
        self.assertEqual(len(model.calls), 4)
        self.assertEqual([x[0] for x in model.calls],
                         ["planner", "researcher", "verifier", "reporter"])
        self.assertEqual(len(report.claims), 1)
        self.assertEqual(report.claims[0].claim_id, claim_id)
        self.assertTrue(all(model.calls[i][1] == INSTRUCTIONS[role]
                            for i, role in enumerate(("planner", "researcher", "verifier", "reporter"))))

    def test_researcher_fabricated_citations_are_rejected(self):
        model = StubModel({
            "planner": {"queries": ["login investigation"]},
            "researcher": {"candidates": [
                {"document_id": "INC-101", "quote": "Totally invented sentence."},
                {"document_id": "NONEXISTENT", "quote": "Fake quote"},
            ]},
            "verifier": {"accepted_claim_ids": []},
            "reporter": {"selected_claim_ids": []},
        })
        report = MultiAgentEngine(self.kb, model=model).run(QUESTION)
        self.assertEqual(report.rejected_proposals, 2)
        self.assertEqual(report.claims, ())
        self.assertIn("No source-verified", report.answer)
        self.assertEqual(report.metrics["model_calls"], 4)

    def test_researcher_unauthorized_schema_rejected(self):
        model = StubModel({
            "planner": {"queries": ["login"]},
            "researcher": {"candidates": [], "execute": "disable_all"},
        })
        with self.assertRaisesRegex(ValueError, "researcher output schema"):
            MultiAgentEngine(self.kb, model=model).run(QUESTION)

    def test_verifier_cannot_approve_invented_ids(self):
        quote = "The recorded successful administrator login occurred during that maintenance window."
        model = StubModel({
            "planner": {"queries": ["approved maintenance login"]},
            "researcher": {"candidates": [{"document_id": "INC-102", "quote": quote}]},
            "verifier": {"accepted_claim_ids": ["CLM-FAKE"]},
        })
        with self.assertRaisesRegex(ValueError, "unauthorized claim"):
            MultiAgentEngine(self.kb, model=model).run(QUESTION)

    def test_reporter_cannot_insert_unverified_ids(self):
        quote = "The recorded successful administrator login occurred during that maintenance window."
        cid = "CLM-" + digest(["INC-102", quote])[:12].upper()
        model = StubModel({
            "planner": {"queries": ["approved maintenance login"]},
            "researcher": {"candidates": [{"document_id": "INC-102", "quote": quote}]},
            "verifier": {"accepted_claim_ids": [cid]},
            "reporter": {"selected_claim_ids": ["CLM-HALLUCINATED"]},
        })
        with self.assertRaisesRegex(ValueError, "unauthorized claim"):
            MultiAgentEngine(self.kb, model=model).run(QUESTION)

    def test_reporter_can_omit_all_claims(self):
        quote = "The recorded successful administrator login occurred during that maintenance window."
        cid = "CLM-" + digest(["INC-102", quote])[:12].upper()
        model = StubModel({
            "planner": {"queries": ["approved maintenance login"]},
            "researcher": {"candidates": [{"document_id": "INC-102", "quote": quote}]},
            "verifier": {"accepted_claim_ids": [cid]},
            "reporter": {"selected_claim_ids": []},
        })
        report = MultiAgentEngine(self.kb, model=model).run(QUESTION)
        self.assertEqual(report.claims, ())

    def test_model_role_with_malformed_output_rejected(self):
        model = StubModel({"planner": ["just a list"]})
        with self.assertRaisesRegex(ValueError, "invalid JSON"):
            MultiAgentEngine(self.kb, model=model).run(QUESTION)

    def test_read_tool_only_reads_known_document_ids(self):
        tools = ReadOnlyTools(self.kb, AuditTrail())
        with self.assertRaisesRegex(ValueError, "unknown"):
            tools.invoke("read_document", document_id="../../credentials")

    def test_audit_tampering_detected(self):
        report = MultiAgentEngine(self.kb).run(QUESTION)
        entries = [dict(row) for row in report.audit]
        self.assertTrue(AuditTrail.verify(entries))
        entries[0]["action"] = "wrong"
        self.assertFalse(AuditTrail.verify(entries))

    def test_claim_ids_stable(self):
        a = MultiAgentEngine(self.kb).run(QUESTION)
        b = MultiAgentEngine(self.kb).run(QUESTION)
        self.assertEqual([c.claim_id for c in a.claims], [c.claim_id for c in b.claims])


if __name__ == "__main__":
    unittest.main()
