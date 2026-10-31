import io
import json
import unittest
from unittest.mock import patch

from agent_forge_lab.providers import OpenAIResponsesModel


class Response:
    def __init__(self, content):
        self.stream = io.BytesIO(content)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, n=-1):
        return self.stream.read(n)


def pack(text, status="completed"):
    return json.dumps({"status": status, "output": [
        {"type": "message", "content": [{"type": "output_text", "text": text}]}
    ]}).encode("utf-8")


class ProviderTests(unittest.TestCase):
    def test_opt_in_is_required_even_if_key_present(self):
        with self.assertRaises(PermissionError):
            OpenAIResponsesModel(key="fake", model="example")

    def test_empty_api_key_rejected(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": ""}):
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                OpenAIResponsesModel(allow_external_data_transfer=True, model="example")

    def test_bad_model_id_rejected(self):
        with self.assertRaisesRegex(ValueError, "model ID"):
            OpenAIResponsesModel(allow_external_data_transfer=True,
                                 model="unknown model with spaces", key="fake")

    def test_bad_timeout_rejected(self):
        for timeout in (0, 121, True):
            with self.subTest(timeout=timeout):
                with self.assertRaisesRegex(ValueError, "timeout"):
                    OpenAIResponsesModel(allow_external_data_transfer=True,
                                         model="example", key="fake", timeout=timeout)

    def test_four_roles_reject_unknown_role(self):
        provider = OpenAIResponsesModel(allow_external_data_transfer=True,
                                        model="example", key="fake")
        with self.assertRaisesRegex(ValueError, "unknown model role"):
            provider.ask("shell", "do things", {})

    def test_mock_response_and_store_false(self):
        body = pack('{"queries":["login"]}')
        with patch("agent_forge_lab.providers.request.urlopen", return_value=Response(body)) as call:
            provider = OpenAIResponsesModel(allow_external_data_transfer=True,
                                            model="example", key="fake")
            result = provider.ask("planner", "Be a planner", {"question": "login"})
        self.assertEqual(result, {"queries": ["login"]})
        req = call.call_args.args[0]
        body = json.loads(req.data)
        self.assertFalse(body["store"])
        self.assertEqual(body["model"], "example")
        self.assertEqual(req.full_url, "https://api.openai.com/v1/responses")
        self.assertEqual(call.call_args.kwargs["timeout"], 20)

    def test_incomplete_response_rejected(self):
        with patch("agent_forge_lab.providers.request.urlopen",
                   return_value=Response(pack("{}", "incomplete"))):
            p = OpenAIResponsesModel(allow_external_data_transfer=True,
                                     model="example", key="fake")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                p.ask("planner", "x", {})

    def test_json_array_response_rejected(self):
        with patch("agent_forge_lab.providers.request.urlopen",
                   return_value=Response(pack("[]"))):
            p = OpenAIResponsesModel(allow_external_data_transfer=True,
                                     model="example", key="fake")
            with self.assertRaisesRegex(ValueError, "JSON object"):
                p.ask("planner", "x", {})

    def test_response_too_large_rejected(self):
        with patch("agent_forge_lab.providers.request.urlopen",
                   return_value=Response(b"X" * 256001)):
            p = OpenAIResponsesModel(allow_external_data_transfer=True,
                                     model="example", key="fake")
            with self.assertRaisesRegex(ValueError, "size cap"):
                p.ask("planner", "x", {})

    def test_response_invalid_json_rejected(self):
        with patch("agent_forge_lab.providers.request.urlopen",
                   return_value=Response(pack("not-json"))):
            p = OpenAIResponsesModel(allow_external_data_transfer=True,
                                     model="example", key="fake")
            with self.assertRaises(json.JSONDecodeError):
                p.ask("planner", "x", {})


if __name__ == "__main__":
    unittest.main()
