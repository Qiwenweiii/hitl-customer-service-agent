"""会话归属切片测试：SessionStore 的单元行为 + /api/chat 的会话归属接线。

单元层证明存储类本身（issue/verify/has 与失败路径）。
HTTP 层用假 Agent（原样记录 invoke 是否被调用）证明分支规则：
A 不带 thread_id → 新建即签发；B 未知 thread → 403 且绝不签发；
C 凭据缺失/不符 → 403 且绝不轮换已有 token；D 校验通过才调用 Agent。
真图假模型与应用工厂两层沿用 test_chat_api_slice.py 的既有覆盖，本轮不重复。
"""

import re

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.api.chat import SESSION_FORBIDDEN_DETAIL, create_chat_app
from app.session_store import SessionStore

CHAT_URL = "/api/chat"
ANSWER = "会话归属校验通过后的答复。"
THREAD = "chat-abcdef123456"
OTHER_THREAD = "chat-000000000000"

# secrets.token_urlsafe 的取值域：URL-safe Base64 字符集，无填充。
URLSAFE_TOKEN_PATTERN = re.compile(r"^[A-Za-z0-9_-]{40,}$")


@pytest.fixture()
def store() -> SessionStore:
    return SessionStore()


class RecordingAgent:
    """假 Agent：记录每次 invoke 的入参；被调用与否就是「有没有放行到 Agent 层」的判据。"""

    def __init__(self) -> None:
        self.calls: list[tuple[object, object]] = []

    def invoke(self, payload: object, config: object = None) -> dict:
        self.calls.append((payload, config))
        return {"messages": [AIMessage(content=ANSWER)]}


@pytest.fixture()
def agent() -> RecordingAgent:
    return RecordingAgent()


@pytest.fixture()
def client(store: SessionStore, agent: RecordingAgent) -> TestClient:
    return TestClient(create_chat_app(agent, store))


def _create_session(client: TestClient) -> tuple[str, str]:
    """走分支 A 新建一个会话，返回 (thread_id, session_token)。"""
    response = client.post(CHAT_URL, json={"message": "你好"})
    assert response.status_code == 200, response.text
    body = response.json()
    return body["thread_id"], body["session_token"]


def test_issue_returns_plaintext_token_and_verify_passes(store):
    token = store.issue(THREAD)
    assert isinstance(token, str) and token
    assert store.verify(THREAD, token) is True


def test_verify_rejects_wrong_token(store):
    store.issue(THREAD)
    assert store.verify(THREAD, "not-the-issued-token") is False


def test_verify_rejects_none_token(store):
    store.issue(THREAD)
    assert store.verify(THREAD, None) is False


def test_verify_rejects_unknown_thread(store):
    token = store.issue(THREAD)
    # token 本身有效，但不属于这个从未签发过的 thread。
    assert store.verify(OTHER_THREAD, token) is False


def test_has_reflects_issue(store):
    assert store.has(THREAD) is False
    store.issue(THREAD)
    assert store.has(THREAD) is True
    # 会话之间互不影响：别的 thread 签发与否不改变本 thread 的状态。
    assert store.has(OTHER_THREAD) is False


def test_reissue_rotates_token_and_invalidates_old(store):
    old = store.issue(THREAD)
    new = store.issue(THREAD)
    assert old != new
    assert store.verify(THREAD, old) is False
    assert store.verify(THREAD, new) is True


def test_threads_have_independent_sessions(store):
    token_a = store.issue(THREAD)
    token_b = store.issue(OTHER_THREAD)
    assert token_a != token_b
    assert store.verify(THREAD, token_b) is False
    assert store.verify(OTHER_THREAD, token_a) is False


# ---------------------------------------------------------------- HTTP 层：/api/chat 归属接线


def test_t1_new_session_issues_token_in_response_body(client, agent, store):
    """分支 A：不带 thread_id 新建会话，响应体回显合法凭据并真的存入 store。"""
    response = client.post(CHAT_URL, json={"message": "你好"})
    assert response.status_code == 200, response.text
    body = response.json()
    token = body["session_token"]
    assert body["thread_id"].startswith("chat-")
    assert URLSAFE_TOKEN_PATTERN.match(token), f"token 不是合法的 urlsafe 串：{token!r}"
    assert store.verify(body["thread_id"], token) is True
    assert len(agent.calls) == 1


def test_t2_correct_thread_and_token_reaches_agent(client, agent):
    """分支 D：正确 thread_id + 正确 token 才放行到 Agent。"""
    thread_id, token = _create_session(client)
    agent.calls.clear()
    response = client.post(CHAT_URL, json={"message": "再问一次", "thread_id": thread_id, "session_token": token})
    assert response.status_code == 200, response.text
    assert len(agent.calls) == 1
    assert agent.calls[0][1] == {"configurable": {"thread_id": thread_id}}


def test_t3_wrong_token_is_forbidden_before_agent(client, agent):
    """分支 C：已知 thread + 错误 token → 统一 403，Agent 一次都不该被调用。"""
    thread_id, _ = _create_session(client)
    agent.calls.clear()
    response = client.post(
        CHAT_URL, json={"message": "你好", "thread_id": thread_id, "session_token": "stolen-or-guess"}
    )
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == SESSION_FORBIDDEN_DETAIL
    assert agent.calls == []


def test_t4_missing_token_is_forbidden_before_agent(client, agent):
    """分支 C：已知 thread + 缺失 token → 同一 403 文案，Agent 不被调用。"""
    thread_id, _ = _create_session(client)
    agent.calls.clear()
    response = client.post(CHAT_URL, json={"message": "你好", "thread_id": thread_id})
    assert response.status_code == 403, response.text
    assert response.json()["detail"] == SESSION_FORBIDDEN_DETAIL
    assert agent.calls == []


def test_t5_failed_attempt_does_not_rotate_the_valid_token(client, agent, store):
    """防接管的关键用例：错误凭据被拒后，原正确 token 必须照常可用。

    若实现把「校验失败」当作重新签发的时机，攻击者的失败请求会顶掉真用户的凭据——
    本用例断言失败尝试前后，旧 token 始终有效且 store 里的哈希一字未动。
    """
    thread_id, token = _create_session(client)
    agent.calls.clear()

    rejected = client.post(
        CHAT_URL, json={"message": "你好", "thread_id": thread_id, "session_token": "wrong-token"}
    )
    assert rejected.status_code == 403, rejected.text

    again = client.post(
        CHAT_URL, json={"message": "你好", "thread_id": thread_id, "session_token": token}
    )
    assert again.status_code == 200, again.text
    assert again.json()["answer"] == ANSWER
    # 轮换没有发生：原 token 仍在 store 里有效。
    assert store.verify(thread_id, token) is True


def test_t6_unknown_thread_is_forbidden_and_never_auto_issued(client, agent, store):
    """分支 B：从未签发过的 thread_id → 403，且不自动签发（响应不含 session_token）。"""
    response = client.post(CHAT_URL, json={"message": "你好", "thread_id": "never-issued-thread"})
    assert response.status_code == 403, response.text
    body = response.json()
    assert body["detail"] == SESSION_FORBIDDEN_DETAIL
    assert "session_token" not in body
    assert "session_token" not in response.headers
    assert store.has("never-issued-thread") is False
    assert agent.calls == []


def test_t7_unknown_thread_attempt_does_not_block_new_session(client, agent):
    """分支 B 被拒后，走分支 A 新建会话照常签发新凭据。"""
    rejected = client.post(CHAT_URL, json={"message": "你好", "thread_id": "never-issued-thread"})
    assert rejected.status_code == 403, rejected.text
    agent.calls.clear()

    created = client.post(CHAT_URL, json={"message": "你好"})
    assert created.status_code == 200, created.text
    body = created.json()
    assert body["thread_id"] != "never-issued-thread"
    assert URLSAFE_TOKEN_PATTERN.match(body["session_token"])
    assert len(agent.calls) == 1


def test_blank_session_token_is_rejected_with_422(client, agent):
    """空白凭据在字段校验层就拦下，不进归属判定，也不消耗 Agent 调用。"""
    thread_id, _ = _create_session(client)
    agent.calls.clear()
    response = client.post(
        CHAT_URL, json={"message": "你好", "thread_id": thread_id, "session_token": "   "}
    )
    assert response.status_code == 422, response.text
    assert agent.calls == []
