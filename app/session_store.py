"""会话归属存储：维护 thread_id -> sha256(session_token) 的映射，为 /api/chat 的会话所有权校验提供底座。

设计边界：
- 公共接口只有 ``issue`` / ``verify`` / ``has`` 三个方法，调用处不直接接触内部存储结构；
  读写都收在 ``_save_hash`` / ``_load_hash`` / ``_contains`` 三个私有方法里（当前实现为进程内 dict），
  之后换 SQLite 只需替换这三个方法的实现，接口签名与调用处不变。
- 只存 token 的 SHA-256 哈希，不落明文；校验时用 ``hmac.compare_digest`` 定时比较，
  避免通过响应时间逐位猜哈希。
- ``create_app()`` 启动时创建一次、所有请求共用同一个实例，且 FastAPI 同步端点跑在线程池里，
  因此所有读写都在 ``threading.Lock`` 下进行。
- issue 对同一 thread_id 重复调用会轮换 token：旧 token 立即失效（存储只保留最新一份哈希）。

本模块只回答「这个 token 是否属于这个 thread」，不产生 thread_id、不接管路由——
接入 HTTP 层是后续切片的事，本轮不动 app/api/、app/main.py 与 Tool 层。
"""

# hashlib：对 token 做单向摘要；hmac：只用它的 compare_digest（定时比较），不是算签名；
# secrets：密码学安全随机源（底层 os.urandom），刻意不用 random / uuid4 / 时间戳拼接；
# threading：一个 store 实例被 FastAPI 线程池里的多个请求线程并发共享。
import hashlib
import hmac
import secrets
import threading

# secrets.token_urlsafe 的字节数：32 字节 ≈ 43 个 URL-safe 字符，熵足够，无需再自己拼随机源。
TOKEN_BYTES = 32


def _hash_token(token: str) -> str:
    """token 明文 -> SHA-256 十六进制摘要；校验双方都走同一实现，避免编码口径漂移。"""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class SessionStore:
    """thread_id 与会话 token 哈希的映射表，线程安全。"""

    def __init__(self) -> None:
        # 锁是实例级而非模块全局：与「create_app() 一个进程只建一个 store」的装配决定配套。
        self._lock = threading.Lock()
        # 唯一状态。value 只存哈希，任何地方都拿不回明文——这是 issue 的
        # 「明文仅此一次、之后只能重新签发」语义的实现基础。
        self._hash_by_thread: dict[str, str] = {}

    def issue(self, thread_id: str) -> str:
        """为会话签发新 token，返回明文（仅此一次，之后无法找回，只能重新签发）。

        同一 thread_id 重复签发即轮换：新哈希覆盖旧哈希，旧 token 从此校验不通过。
        """
        # 顺序是刻意的：随机数生成（慢）放锁外，锁只保护 dict 写入这一小段临界区。
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with self._lock:
            self._save_hash(thread_id, _hash_token(token))
        # 调用者唯一一次看到明文的时机。正常流程里本方法只会被路由的「新建会话」分支触达
        # （app/api/chat.py 分支 A，由守卫测试钉住）；校验失败的路径绝不许走到这里，
        # 否则攻击者试几次错 token 就能把合法持有者踢下线（轮换 = 劫持）。
        return token

    def verify(self, thread_id: str, token: str | None) -> bool:
        """校验 token 是否属于该 thread；thread 未签发过或 token 为 None 一律 False。"""
        # 三道判定按「越靠前越省」排序。签名接受 None 是让调用处把「请求里没带 token」
        # 原样传进来，不必先自行判空。
        if token is None:
            return False
        with self._lock:
            stored = self._load_hash(thread_id)
            if stored is None:
                return False
        # 定时比较放在锁外：哈希计算与比较都不触碰共享状态，锁早放减少并发争用。
        # 用 compare_digest 而不是 ==：普通字符串比较不匹配时提前返回，耗时随「前几位
        # 相同多少」变化，理论上可被计时攻击逐位猜哈希；compare_digest 耗时与内容无关。
        # 对「未知 thread」与「错 token」本方法都返回 False——错误形态相同，
        # 不给外部留下探测 thread 是否存在的缝隙（路由层据此统一回 403）。
        return hmac.compare_digest(stored, _hash_token(token))

    def has(self, thread_id: str) -> bool:
        """该 thread 是否已签发过会话 token。"""
        # 只答「签发过没有」。verify 对未知 thread 与错 token 都返回 False，
        # 路由层需要这个独立问题来短路区分（app/api/chat.py:252 的 not has or not verify），
        # 但两者对外仍是同一句 403，探测价值被抹平。
        with self._lock:
            return self._contains(thread_id)

    # 存储读写只经这三个私有方法暴露给类内部；换 SQLite 时改这里，不动上面的公共接口。
    # 对应关系：_save_hash -> INSERT OR REPLACE，_load_hash -> SELECT，_contains -> SELECT 1。

    def _save_hash(self, thread_id: str, token_hash: str) -> None:
        self._hash_by_thread[thread_id] = token_hash

    def _load_hash(self, thread_id: str) -> str | None:
        return self._hash_by_thread.get(thread_id)

    def _contains(self, thread_id: str) -> bool:
        return thread_id in self._hash_by_thread
