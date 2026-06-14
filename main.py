import gc

from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star


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
    """给 LLM 一个「我说完了」工具：在没有更多内容要输出时主动、干净地结束本回合。

    背景：AstrBot 的 agent 工具循环里，如果模型在调用工具之后那一轮交了空回复
    （无正文、无思考、无工具调用），provider 会判为「无输出」并重试 3 次、最终报错降级。
    本插件提供 end_reply 工具——它返回 None，触发 runner 的 `resp is None` 分支
    （AgentState.DONE），让模型把「不再输出」变成一次主动、合法的结束，而不是故障。

    注意：runner 的 `resp is None` 分支只转 DONE、从不设置 final_llm_resp，
    导致管线 _save_to_history 因 final_resp 为 None 而跳过保存、本回合不进会话记录。
    因此本插件在 return None 之前，用 gc 反查到当前 runner 并补一个非 None 的
    assistant 响应，让保存逻辑放行（真正写入的历史仍是 run_context.messages，
    这个值只用于「放行保存」，正文留空即可）。
    """

    def __init__(self, context: Context):
        super().__init__(context)

    @filter.llm_tool(name="end_reply")
    async def end_reply(self, event: AstrMessageEvent):
        """当你这一轮已经把想说的都表达完、没有更多内容要输出时，调用本工具干净地结束本回合。
        用它来代替「交一条空回复」——空回复会被系统判为无输出、反复重试并最终报错降级。
        注意：如果你还有话要说，就直接正常写出来，不要调用本工具。
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
