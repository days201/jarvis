"""Exercise streamed replies, cancellation, and the sentence queue offline."""

import json
import copy
import threading
import unittest
from unittest.mock import patch

from assistant import AssistantError, DeepSeekClient, ENDPOINT, SentenceBuffer


class Reply:
    def __init__(self, lines=(), status=200):
        self.lines = lines
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_lines(self, **kwargs):
        yield from self.lines


def event(delta):
    return b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode()


def tool_call(name, arguments, index=0):
    return {"index": index, "id": f"camera_{index}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments)}}


class CloudRobot:
    cloud_vision = True

    def __init__(self):
        self.frames = self
        self.calls = []
        self.snapshots = 0

    def latest(self):
        return True

    def snapshot(self, cancelled):
        self.snapshots += 1
        return "data:image/jpeg;base64,TEST" + str(self.snapshots)

    def context(self):
        return {"camera_online": True}

    def execute(self, name, arguments, cancelled):
        self.calls.append((name, arguments))
        if name == "capture_frame":
            return {"ok": True, "_image": self.snapshot(cancelled)}
        return {"ok": True, "position": [0.1, 0]}


class AssistantTests(unittest.TestCase):
    def run_replies(self, client, responses, text="Look right", **kwargs):
        payloads = []
        responses = iter(responses)
        def post(*args, **options):
            payloads.append(copy.deepcopy(options["json"]))
            return next(responses)
        with patch("assistant.requests.Session") as session:
            session.return_value.__enter__.return_value.post.side_effect = post
            reply = "".join(client.reply(text, threading.Event(), **kwargs))
        return reply, payloads

    def test_stream_ignores_reasoning_and_keeps_context(self):
        client = DeepSeekClient("test-key")
        event = lambda delta: b"data: " + json.dumps({"choices": [{"delta": delta}]}).encode()
        response = Reply([event({"reasoning_content": "not a spoken answer"}), event({"content": "Hello"}), event({"content": " there."}), b"data: [DONE]"])
        with patch("assistant.requests.Session") as session:
            session.return_value.__enter__.return_value.post.return_value = response
            self.assertEqual("".join(client.reply("Hi", threading.Event())), "Hello there.")
            args, kwargs = session.return_value.__enter__.return_value.post.call_args
            self.assertEqual(args[0], ENDPOINT)
            self.assertFalse(kwargs["allow_redirects"])
            self.assertEqual(kwargs["json"]["thinking"], {"type": "disabled"})
        self.assertEqual(client._history[-1]["content"], "Hello there.")

    def test_authentication_error_does_not_expose_key(self):
        client = DeepSeekClient("test-key-do-not-display")
        with patch("assistant.requests.Session") as session:
            session.return_value.__enter__.return_value.post.return_value = Reply(status=401)
            with self.assertRaises(AssistantError) as caught:
                list(client.reply("Hi", threading.Event()))
        self.assertNotIn("test-key", str(caught.exception))
        self.assertNotIn("test-key", repr(client))

    def test_cancelled_stream_is_not_added_to_conversation(self):
        client = DeepSeekClient("test-key")
        cancelled = threading.Event()
        cancelled.set()
        with patch("assistant.requests.Session") as session:
            session.return_value.__enter__.return_value.post.return_value = Reply([b"data: [DONE]"])
            self.assertEqual(list(client.reply("Hi", cancelled)), [])
        self.assertEqual(client._history, [])

    def test_sentence_buffer_preserves_decimal_and_final_words(self):
        buffer = SentenceBuffer()
        self.assertEqual(buffer.add("It costs 2.5 dollars. Next"), ["It costs 2.5 dollars."])
        self.assertEqual(buffer.add(" sentence! "), ["Next sentence!"])
        self.assertEqual(buffer.add("Last words"), [])
        self.assertEqual(buffer.finish(), ["Last words"])

    def test_image_input_is_current_turn_only_and_never_saved_in_history(self):
        client, robot = DeepSeekClient("test-key"), CloudRobot()
        reply, payloads = self.run_replies(client, [Reply([event({"content": "It is blue."}), b"data: [DONE]"])],
                                          text="What color is this?", robot=robot, new_conversation=True)
        self.assertEqual(reply, "It is blue.")
        self.assertEqual(payloads[0]["messages"][-1]["content"][1]["type"], "image_url")
        self.assertEqual(client.metrics["images"], 1)
        self.assertNotIn("data:image", json.dumps(client._history))
        self.assertNotIn("image_url", json.dumps(client._history))

    def test_plain_followup_does_not_upload_another_photo(self):
        robot = CloudRobot()
        self.run_replies(DeepSeekClient("test-key"), [Reply([event({"content": "Four."}), b"data: [DONE]"])],
                         text="What is two plus two?", robot=robot, new_conversation=False)
        self.assertEqual(robot.snapshots, 0)

    def test_fragmented_tool_calls_execute_once_then_get_a_fresh_view(self):
        client, robot = DeepSeekClient("test-key"), CloudRobot()
        first = Reply([event({"tool_calls": [{"index": 0, "id": "action_1", "function": {"name": "lo", "arguments": '{"direction":'}}]}),
                       event({"tool_calls": [{"index": 0, "function": {"name": "ok", "arguments": '"right", "degrees": 10}'}}]}), b"data: [DONE]"])
        second = Reply([event({"content": "I turned right."}), b"data: [DONE]"])
        reply, payloads = self.run_replies(client, [first, second], robot=robot, new_conversation=True)
        self.assertEqual(robot.calls, [("look", {"direction": "right", "degrees": 10})])
        self.assertEqual(reply, "I turned right.")
        self.assertEqual(client.metrics["images"], 2)
        image_urls = [part["image_url"]["url"] for payload in payloads for message in payload["messages"]
                      if isinstance(message.get("content"), list) for part in message["content"] if part["type"] == "image_url"]
        self.assertEqual(image_urls, ["data:image/jpeg;base64,TEST1", "data:image/jpeg;base64,TEST2"])

    def test_incomplete_tool_stream_cannot_move_camera(self):
        robot = CloudRobot()
        with self.assertRaises(AssistantError):
            self.run_replies(DeepSeekClient("test-key"), [Reply([event({"tool_calls": [tool_call("look", {"direction": "left"})]})])], robot=robot)
        self.assertEqual(robot.calls, [])

    def test_two_image_budget_blocks_extra_capture_calls(self):
        client, robot = DeepSeekClient("test-key"), CloudRobot()
        calls = [tool_call("capture_frame", {}, index) for index in range(3)]
        self.run_replies(client, [Reply([event({"tool_calls": calls}), b"data: [DONE]"]),
                                 Reply([event({"content": "Done."}), b"data: [DONE]"])], robot=robot, new_conversation=True)
        self.assertEqual(robot.snapshots, 2)
        self.assertEqual(robot.calls, [("capture_frame", {})])
        self.assertEqual(client.metrics["images"], 2)

    def test_arrival_greeting_never_uploads_a_photo_or_exposes_tools(self):
        robot = CloudRobot()
        _, payloads = self.run_replies(DeepSeekClient("test-key"), [Reply([event({"content": "Hello."}), b"data: [DONE]"])],
                                      text="A person arrived", robot=robot, new_conversation=True, automatic=True)
        self.assertNotIn("tools", payloads[0])
        self.assertEqual(robot.snapshots, 0)
        self.assertEqual(robot.calls, [])

    def test_action_speech_is_initial_acknowledgement_then_final_result(self):
        spoken = []
        responses = [
            Reply([event({'content': 'I will center you.'}),
                   event({'tool_calls': [tool_call('look_at', {'x': 0.4, 'y': 0.6})]}), b'data: [DONE]']),
            Reply([event({'content': 'You are still at the edge. One more adjustment.'}),
                   event({'tool_calls': [tool_call('look_at', {'x': 0.5, 'y': 0.6})]}), b'data: [DONE]']),
            Reply([event({'content': 'You are centered. Following is off.'}), b'data: [DONE]'])]
        text, _ = self.run_replies(DeepSeekClient('test-key'), responses,
                                   robot=CloudRobot(), on_speech=spoken.append)
        self.assertEqual(''.join(spoken), 'I will center you.\nYou are centered. Following is off.')
        self.assertIn('One more adjustment.', text)

    def test_ordinary_speech_streams_before_model_finishes(self):
        seen = []
        def lines():
            yield event({'content': 'First sentence. '})
            self.assertEqual(seen, ['First sentence. '])
            yield event({'content': 'Second sentence.'})
            yield b'data: [DONE]'
        self.run_replies(DeepSeekClient('test-key'), [Reply(lines())], on_speech=seen.append)
        self.assertEqual(''.join(seen), 'First sentence. Second sentence.')


if __name__ == "__main__":
    unittest.main()
