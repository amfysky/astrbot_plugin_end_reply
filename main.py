import gc

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

# 预注入给 LLM 的规则：纯文本正文会被丢弃，必须用 send_message_to_user 工具发消息。
# 用一个稳定的标记前缀来做幂等，避免 agent 多轮里被重复追加。
_INJECT_MARK = "【消息发送规则】"
_INJECT_NOTE = (
    "\n\n"
    + _INJECT_MARK
    + "这是必须严格遵守的硬性规则：\n"
    "1. 你【唯一】能让用户看到内容的方式，是调用 `send_message_to_user` 工具(tool)，"
    "把要对用户说的话作为参数传给它。\n"
    "2. 你直接写出的任何纯文本正文都【不会】发送给用户——系统会在下发前把模型直接回复里的"
    "所有文本丢弃，用户根本看不到。\n"
    "3. 因此：不要把回复直接写成正文，也不要在工具调用之外输出文字。每一次想对用户说话，"
    "都【必须】通过调用 `send_message_to_user` 来完成；不调用该工具，就等于什么都没说。\n"
    "4. 当你本轮没有更多内容要发送时，调用 `end_reply` 工具(tool)干净地结束本回合，"
    "不要交一条空回复（空回复会被判为无输出、反复重试并最终报错）。"
)


def _is_llm_result(result) -> bool:
    """该结果是否为「大模型直接回复」(LLM_RESULT)，而非命令/工具直发等。

    只拦截这一类结果，命令回复、工具用 event.send 直发的内容都不是 LLM_RESULT，
    天然不受影响。
    """
    fn = getattr(result, "is_llm_result", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            pass
    # 兜底：按 result_content_type 的枚举名判断
    rct = getattr(result, "result_content_type", None)
    return getattr(rct, "name", None) == "LLM_RESULT"


def _strip_plain(chain) -> None:
    """就地剔除组件列表里的所有文本(Plain)组件，保留图片等非文本组件。"""
    if not isinstance(chain, list):
        return
    try:
        from astrbot.api.message_components import Plain
    except Exception:
        return
    kept = [c for c in chain if not isinstance(c, Plain)]
    if len(kept) != len(chain):
        chain[:] = kept


def _find_running_runner(event: AstrMessageEvent):
    """通过 gc 反查正在驱动本次 event 的本地 Agent runner。"""
    try:
        from astrbot.core.agent.runners.tool_loop_agent_runner import (
            ToolLoopAgentRunner,
        )
    except Exception:
        return None
    for obj in gc.get_objects():
        if isinstance(obj, ToolLoopAgentRunner):
            rc = getattr(obj, "run_context", None)
            ctx = getattr(rc, "context", None)
            # 按 event 身份精确匹配，天然隔离并发会话
            if ctx is not None and getattr(ctx, "event", None) is event:
                return obj
    return None


class BlockReplyPlugin(Star):
    """屏蔽回复：拦截并丢弃模型的一切纯文本输出，使其只能通过「工具」向用户发消息。

    一、拦截文本输出（on_decorating_result）
    AstrBot 会把模型「不带工具调用的那一段正文」当作普通回复自动下发：在本地 Agent
    模式下，run_agent 把这段正文 set_result 成 ResultContentType.LLM_RESULT，随后管线
    的结果装饰阶段(ResultDecorateStage)→发送阶段(RespondStage)把它发出去。
    本插件在结果装饰阶段(on_decorating_result)介入：当结果是「大模型直接回复」
    (is_llm_result) 时，剔除其消息链里的全部文本(Plain)组件。链被清空后发送阶段判定
    为空消息、不再下发，于是模型「直接写正文」这条路被彻底掐断，内容只能经由工具(tool)
    发送（工具直发走 event.send，不经过本阶段；命令回复等不是 LLM_RESULT，均不受影响）。
    注意：这里只拦「发送」，历史记录里仍保留模型原始正文（_save_to_history 用的是
    run_context.messages，与本阶段无关）。

    二、预注入规则（on_llm_request）
    在 on_llm_request 钩子里把上述约定追加进 system_prompt，明确告诉模型：纯文本
    会被丢弃、只能用工具发消息、说完用 end_reply 收尾。带标记前缀做幂等，避免
    agent 多轮里重复追加。

    三、主动结束（end_reply 工具）
    背景：AstrBot 的 agent 工具循环里，如果模型在调用工具之后那一轮交了空回复
    （无正文、无思考、无工具调用），provider 会判为「无输出」并重试 3 次、最终报错降级。
    end_reply 工具返回 None，触发 runner 的 `resp is None` 分支（AgentState.DONE），
    让模型把「不再输出」变成一次主动、合法的结束，而不是故障。

    注意：runner 的 `resp is None` 分支只转 DONE、从不设置 final_llm_resp，
    导致管线 _save_to_history 因 final_resp 为 None 而跳过保存、本回合不进会话记录。
    因此本插件在 return None 之前，用 gc 反查到当前 runner 并补一个非 None 的
    assistant 响应，让保存逻辑放行（真正写入的历史仍是 run_context.messages，
    这个值只用于「放行保存」，正文留空即可）。

    局限：流式输出(streaming_response)下，正文是边生成边逐块下发的，在结果装饰阶段
    之前就已发出，本拦截对流式模式无效。如需在流式下生效，请关闭流式输出。
    """

    def __init__(self, context: Context):
        super().__init__(context)

    @filter.on_llm_request()
    async def announce_text_discarded(self, event: AstrMessageEvent, req):
        """预注入：把「纯文本会被丢弃、只能用工具发消息」这条规则告诉 LLM。"""
        try:
            sp = req.system_prompt or ""
            if _INJECT_MARK not in sp:
                req.system_prompt = sp + _INJECT_NOTE
        except Exception:
            # 即使 ProviderRequest 结构有变，也不影响其余功能
            pass

    @filter.on_decorating_result()
    async def discard_model_text(self, event: AstrMessageEvent):
        """拦截模型的一切文本输出：在消息下发前，把「大模型直接回复」结果里的
        文本(Plain)组件全部剔除，使模型无法靠直接写正文向用户发消息——内容只能
        经由工具(tool)发送。命令回复、工具直发等不是 LLM_RESULT，不受影响。
        """
        try:
            result = event.get_result()
            if result is None or not getattr(result, "chain", None):
                return
            if not _is_llm_result(result):
                return
            _strip_plain(result.chain)
        except Exception:
            # 反射兜底：内部结构变了也不影响插件其余功能
            pass

    @filter.llm_tool(name="end_reply")
    async def end_reply(self, event: AstrMessageEvent):
        """当你这一轮已经把想说的都用「消息发送」工具发完、没有更多内容要发送时，调用本工具干净地结束本回合。
        用它来代替「交一条空回复」——空回复会被系统判为无输出、反复重试并最终报错降级。
        注意：你直接写出的纯文本不会发给用户（会被丢弃），要发内容请调用消息发送工具，而不是调用本工具。
        """
        # 在 return None 之前补设 final_llm_resp，让本回合能正常写入会话记录。
        try:
            runner = _find_running_runner(event)
            if runner is not None and runner.get_final_llm_resp() is None:
                from astrbot.core.provider.entities import LLMResponse

                runner.final_llm_resp = LLMResponse(
                    role="assistant", completion_text=""
                )
        except Exception:
            # 反射兜底：即使内部结构变了也不影响工具本身正常结束
            pass

        # 返回 None → 工具执行器 yield None → agent 循环转入 DONE，不再请求「最后一轮」。
        # 不调用 stop_event()，以免误伤同一轮里模型已写出的正文、并卡住保存。
        return None
