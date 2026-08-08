"""Voice/NLU layer tests.

These run fully offline — they exercise the keyword fallback and the transcript
filter, never the Ollama server or a microphone, so CI needs no services.
The LLM path is covered by a schema test that validates whatever it returns.
"""

import numpy as np
import pytest

import nlu
import voice


# ------------------------------------------------------------ intent parsing
@pytest.mark.parametrize("phrase,action,trick", [
    ("walk forward",          "move",  ""),
    ("go fast",               "move",  ""),
    ("stop",                  "move",  ""),
    ("back up",               "move",  ""),
    ("sit down",              "trick", "sit"),
    ("lie down",              "trick", "lie"),
    ("wave hello",            "trick", "wave"),
    ("can you dance",         "trick", "dance"),
    ("take a bow",            "trick", "bow"),
    ("spin around",           "trick", "spin"),
    ("go fetch the ball",     "trick", "fetch"),
    ("patrol the arena",      "trick", "patrol"),
    ("follow the ball",       "trick", "follow"),
])
def test_keyword_fallback_maps_intents(phrase, action, trick):
    """The offline fallback must handle every command the robot supports —
    it is the safety net whenever Ollama is down."""
    out = nlu._keyword_fallback(phrase, 0.0, 0.0)
    assert out["action"] == action, f"{phrase!r} -> {out['action']}, want {action}"
    if trick:
        assert out["trick"] == trick


def test_fallback_turn_directions_are_not_mirrored():
    """Left must be +yaw and right -yaw. Getting this backwards is the kind of
    bug that is obvious in a demo and invisible in code review."""
    assert nlu._keyword_fallback("turn left", 0, 0)["yaw"] > 0
    assert nlu._keyword_fallback("turn right", 0, 0)["yaw"] < 0


def test_fallback_relative_speed_uses_current_state():
    faster = nlu._keyword_fallback("go faster", 0.3, 0.0)
    slower = nlu._keyword_fallback("slow down", 0.3, 0.0)
    assert faster["vx"] > 0.3 and slower["vx"] < 0.3


def test_fallback_never_exceeds_command_limits():
    for phrase, cur in [("go faster", 0.6), ("faster", 5.0), ("slow down", -9.0)]:
        out = nlu._keyword_fallback(phrase, cur, 0.0)
        assert nlu.VX_LO <= out["vx"] <= nlu.VX_HI, f"{phrase} escaped vx limits"
        assert nlu.YAW_LO <= out["yaw"] <= nlu.YAW_HI


@pytest.mark.parametrize("phrase", [
    "", "   ", "asdfghjkl", "what is the weather", "!!!", "1234",
])
def test_unparseable_input_still_returns_valid_intent(phrase):
    """Garbage in must never crash the voice loop mid-conversation."""
    out = nlu._keyword_fallback(phrase, 0.2, 0.1)
    _assert_valid_intent(out)


def _assert_valid_intent(out):
    assert set(out) == {"action", "vx", "yaw", "trick", "reply"}
    assert out["action"] in ("move", "trick", "chat")
    assert nlu.VX_LO <= out["vx"] <= nlu.VX_HI
    assert nlu.YAW_LO <= out["yaw"] <= nlu.YAW_HI
    assert isinstance(out["reply"], str) and out["reply"]
    if out["action"] == "trick":
        assert out["trick"] in nlu.TRICK_NAMES


def test_every_advertised_trick_is_dispatchable():
    """Any trick the LLM may emit must exist in the trick engine or be a
    controller mode — otherwise the robot silently ignores the user."""
    import tricks
    controller_modes = {"fetch", "patrol", "follow", "spin"}
    for name in nlu.TRICK_NAMES:
        assert name in tricks.TRICKS or name in controller_modes, \
            f"NLU can emit {name!r} but nothing handles it"


def test_parse_command_falls_back_when_llm_unavailable(monkeypatch):
    """If Ollama raises, we must degrade to keywords, not crash."""
    def boom(*a, **k):
        raise ConnectionError("ollama is down")
    monkeypatch.setattr(nlu.ollama, "chat", boom)
    out = nlu.parse_command("sit down")
    _assert_valid_intent(out)
    assert out["action"] == "trick" and out["trick"] == "sit"


def test_parse_command_survives_malformed_llm_json(monkeypatch):
    monkeypatch.setattr(nlu.ollama, "chat",
                        lambda *a, **k: {"message": {"content": "not json at all"}})
    _assert_valid_intent(nlu.parse_command("walk forward"))


def test_parse_command_clamps_out_of_range_llm_output(monkeypatch):
    """A hallucinated vx=99 must be clamped, never sent to the robot."""
    monkeypatch.setattr(nlu.ollama, "chat", lambda *a, **k: {"message": {"content":
        '{"action":"move","vx":99,"yaw":-42,"trick":"","reply":"zoom"}'}})
    out = nlu.parse_command("go")
    assert out["vx"] == nlu.VX_HI and out["yaw"] == nlu.YAW_LO


def test_parse_command_rejects_invented_trick(monkeypatch):
    monkeypatch.setattr(nlu.ollama, "chat", lambda *a, **k: {"message": {"content":
        '{"action":"trick","vx":0,"yaw":0,"trick":"backflip","reply":"ok"}'}})
    out = nlu.parse_command("do a backflip")
    assert out["action"] == "chat", "unknown trick must not be dispatched"


# --------------------------------------------------- speech transcript hygiene
@pytest.mark.parametrize("junk", [
    "The The The The The The",   # observed Whisper silence hallucination
    "Thank you.", "thanks for watching", "you", ".", "", "  ",
])
def test_hallucinated_transcripts_are_dropped(junk):
    assert voice._clean_transcript(junk) == ""


@pytest.mark.parametrize("real", [
    "walk forward", "sit down", "the dog should walk", "turn left please",
])
def test_real_speech_survives_the_filter(real):
    assert voice._clean_transcript(real) == real


def test_filter_keeps_legitimate_repeats():
    """'go go go' is a real (if excitable) command; only single-token spam is
    treated as a hallucination."""
    assert voice._clean_transcript("go go faster") == "go go faster"


# --------------------------------------------------------- conversation memory
def test_memory_records_only_actionable_intents():
    m = nlu.ConversationMemory()
    m.record("who are you", {"action": "chat", "vx": 0, "yaw": 0,
                             "trick": "", "reply": "hi"})
    assert m.last_intent is None, "chat must not become the repeatable command"
    sit = {"action": "trick", "vx": 0, "yaw": 0, "trick": "sit", "reply": "ok"}
    m.record("sit", sit)
    assert m.last_intent["trick"] == "sit"


def test_repeat_phrase_replays_last_command_without_the_llm(monkeypatch):
    """'do it again' must work even with Ollama down."""
    monkeypatch.setattr(nlu.ollama, "chat", lambda *a, **k:
                        (_ for _ in ()).throw(ConnectionError("down")))
    m = nlu.ConversationMemory()
    m.record("dance", {"action": "trick", "vx": 0, "yaw": 0,
                       "trick": "dance", "reply": "dancing"})
    out = nlu.parse_command("do it again", memory=m)
    assert out["action"] == "trick" and out["trick"] == "dance"
    _assert_valid_intent(out)


def test_repeat_without_history_does_not_crash(monkeypatch):
    monkeypatch.setattr(nlu.ollama, "chat", lambda *a, **k:
                        (_ for _ in ()).throw(ConnectionError("down")))
    _assert_valid_intent(nlu.parse_command("again", memory=nlu.ConversationMemory()))


def test_memory_is_bounded():
    m = nlu.ConversationMemory(maxlen=2)
    for i in range(6):
        m.record(f"cmd {i}", {"action": "move", "vx": 0.1, "yaw": 0,
                              "trick": "", "reply": "ok"})
    assert len(m.as_messages()) == 4, "memory must stay bounded (2 turns x 2 msgs)"


def test_memory_history_is_valid_chat_format():
    m = nlu.ConversationMemory()
    m.record("walk", {"action": "move", "vx": 0.4, "yaw": 0, "trick": "", "reply": "ok"})
    for msg in m.as_messages():
        assert set(msg) == {"role", "content"}
        assert msg["role"] in ("user", "assistant")
        assert isinstance(msg["content"], str)


def test_memory_clear_resets_state():
    m = nlu.ConversationMemory()
    m.record("sit", {"action": "trick", "vx": 0, "yaw": 0, "trick": "sit", "reply": "ok"})
    m.clear()
    assert m.last_intent is None and m.as_messages() == []


# -------------------------------------------------------------------- preflight
def test_preflight_missing_policy_gives_actionable_error(tmp_path):
    import preflight
    with pytest.raises(preflight.PreflightError) as exc:
        preflight.check_policy(str(tmp_path))
    msg = str(exc.value)
    assert "train_v2.py" in msg, "error must tell the user how to fix it"


def test_preflight_finds_a_real_policy():
    import preflight
    from conftest import find_policy
    if find_policy() is None:
        pytest.skip("no policy available")
    import os
    d = os.path.dirname(find_policy()[0])
    model, norm = preflight.check_policy(d)
    assert os.path.exists(model + ".zip") and os.path.exists(norm)


def test_preflight_checks_return_message_pairs():
    import preflight
    for check in (preflight.check_ollama(), preflight.check_microphone(),
                  preflight.check_viewer_runtime()):
        ok, msg = check
        assert isinstance(ok, bool) and isinstance(msg, str) and msg
