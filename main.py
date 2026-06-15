import gc

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

# 预注入给 LLM 的规则：纯文本正文会被丢弃，必须用工具发消息。
# 用一个稳定的标记前缀来做幂等，避免 agent 多轮里被重复追加。
_INJECT_MARK = "【消息发送规则】"
_INJECT_NOTE = (
    "\n\n"
    + _INJECT_MARK
    + "你直接写出的纯文本正文不会发送给用户——系统会在下发前丢弃模型产生的所有文本。"
    "若要把内容发送给用户，必须调用相应的「消息发送」工具(tool)来发送，"
    "只有经由工具发送的内容才会真正到达用户。"
    "当你本轮没有更多内容要发送时，调用 end_reply 工具干净地结束本回合，"
    "不要交一条空回复（空回复会被判为无输出、反复重试并最终报错）。"
)


def _strip_plain(chain):
    """就地剔除组件列表里的所有文本(Plain)组件，保留图片等非文本组件。"""
    try:
        from astrbot.api.message_components import Plain
    except Exception:
        return
    if not isinstance(chain, list):
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


class EndReplyPlugin(Star):
    """让 LLM 只能通过「工具」向用户发消息：拦截并丢弃模型的一切纯文本输出，
    同时提供 end_reply 工具让模型在没有更多内容时主动、干净地结束本回合。

    一、拦截文本输出（on_llm_response）
    在 on_llm_response 钩子里清空本次 LLM 响应的正文（completion_text）并剔除
    result_chain 里的文本组件。这样模型「直接写正文回复」这条路被彻底掐断，
    内容只能经由工具(tool)在执行时主动发送（工具内部走 event.send，不经过本钩子，
    因此不受影响）。钩子只对 LLM 响应生效，命令回复、其它插件的输出一律不受影响。

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

    @filter.on_llm_response()
    async def discard_model_text(self, event: AstrMessageEvent, resp):
        """拦截模型的一切文本输出：清空本次 LLM 响应的正文与结果链里的文本组件，
        使模型无法靠「直接写正文」向用户发消息——内容只能经由工具(tool)发送。
        """
        try:
            # 1) 清掉纯文本正文
            if getattr(resp, "completion_text", None):
                resp.completion_text = ""
            # 2) 清掉已构造结果链里的文本组件（保留图片等非文本组件）
            rc = getattr(resp, "result_chain", None)
            if rc is not None:
                # result_chain 可能是 MessageChain（有 .chain）或直接是组件 list
                _strip_plain(getattr(rc, "chain", rc))
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
