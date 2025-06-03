import gradio as gr
import asyncio
from dataclasses import dataclass
from autogen_core import (
    AgentId,
    RoutedAgent,
    MessageContext,
    default_subscription,
    message_handler,
    SingleThreadedAgentRuntime, DefaultTopicId, DefaultSubscription
)
from autogen_core.models import (
    ChatCompletionClient,
    UserMessage,
    SystemMessage
)
from autogen_ext.models.ollama import (
    OllamaChatCompletionClient
)
from pydantic import BaseModel, Field, ValidationError

@dataclass
class JokeGeneratorRequest:
    """ 請求生成笑話的數據類型 """
    round: int = Field(..., description="當前輪次")

class JokeGeneratorResult(BaseModel):
    """ 笑話生成結果 """
    content: str = Field(..., description="生成的笑話內容")

@dataclass
class JokeEvaluationRequest:
    """ 請求評估笑話的數據類型 """
    joke: str
    round: int

class JokeEvaluationResult(BaseModel):
    """ 笑話評估結果 """
    evaluation: int = Field(..., ge=0, le=10, description="幽默程度分數")
    reason: str = Field(..., description="評分理由")

@dataclass
class JokeContestSummaryRequest:
    """ 笑話比賽總結請求 """
    jokes: list[str] = Field(..., description="參賽者的笑話列表")
    joke_eval_score: list[int] = Field(..., description="每個笑話的評分結果")
    joke_eval_reason: list[str] = Field(..., description="每個笑話的評分理由")

class JokeContestConclusion(BaseModel):
    """ 笑話比賽總結 """
    summary: str = Field(..., description="笑話比賽的總結評論")

joke_list = []
joke_eval_list = []

class JokeGeneratorAgent(RoutedAgent):
    def __init__(self, message_queue: asyncio.Queue, model_client: ChatCompletionClient):
        super().__init__("Joke Generator Agent")
        self._message_queue = message_queue
        self._model_client = model_client
        self._system_message = SystemMessage(content="""
你是一個專業的笑匠，你的任務是逗笑每一個人，輸出一個笑話。只輸出笑話內容，不要有其他內容。

# Notes
- 總是用繁體中文回答。
- 盡可能使用 emoji 來表達情感。
- 你會因為評分過低，被羞辱而爆氣說「他媽的」。
- 除非真的覺得評分不公平，否則不要亂生氣。
""")

    @message_handler
    async def handle_joke_request(self, message: JokeGeneratorRequest, ctx: MessageContext) -> None:
        prompt = f"請輸出一個笑話，這是第 {message.round} 輪。"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
            json_output=JokeGeneratorResult
        )
        content = llm_result.content
        try:
            joke_result = JokeGeneratorResult.model_validate_json(content)

            print(f"----- {message.round} ------")
            print(f"生成的笑話: {joke_result.content}")

            joke_list.append(joke_result)
            await self._message_queue.put({"role": "user", "content": joke_result.content})

            await self.publish_message(
                JokeEvaluationRequest(joke=joke_result.content, round=message.round),
                                      DefaultTopicId(type="joke_evaluation", source=ctx.topic_id.source))

        except ValidationError as e:
            print(e)

class JokeEvaluationAgent(RoutedAgent):
    def __init__(self, message_queue: asyncio.Queue, model_client: ChatCompletionClient, max_rounds: int = 3):
        super().__init__("Joke Evaluation Agent")
        self._message_queue = message_queue
        self._max_rounds = max_rounds
        self._model_client = model_client
        self._system_message = SystemMessage(content="""
你是一個冷酷笑匠, 你的任務是做笑話評分。

# Steps
1. 評估笑話的幽默程度和創意。
2. 給予笑話一個分數 (0-10)，0表示完全不好笑，10表示非常好笑。
3. 給予一個簡短的理由，解釋為什麼給予這個分數。

# Output Format
輸出分數，0到10之間的整數值，以及理由。

# Notes
- 不要輸出其他內容，僅限於分數和理由。
- 對於笑話的標準非常嚴苛。
- 如果是不好笑的笑話，會用尖酸刻薄的羞辱。
- 總是用繁體中文回答。
- 盡可能使用 emoji 來表達情感。
- 如果你覺得笑話很糟糕，可以說"比賽結束"，但這很少發生。
- 笑話不要太長，否則會被認為是廢話。
""")

    @message_handler
    async def handle_joke_evaluation(self, message: JokeEvaluationRequest, ctx: MessageContext) -> None:
        prompt = f"請評估這個笑話：{message.joke}，這是第 {message.round} 輪。"
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
            json_output=JokeEvaluationResult
        )
        content = llm_result.content
        try:
            eval_result = JokeEvaluationResult.model_validate_json(content)
            print(f"評分結果: {eval_result.evaluation} 分，理由: {eval_result.reason}")
            joke_eval_list.append(eval_result)
            await self._message_queue.put(
                {"role": "assistant", "content": f"{eval_result.evaluation} 分，理由: {eval_result.reason}"})
            if message.round >= self._max_rounds:
                await self.publish_message(
                    JokeContestSummaryRequest(
                        jokes=[joke.content for joke in joke_list],
                        joke_eval_score=[e.evaluation for e in joke_eval_list],
                        joke_eval_reason=[e.reason for e in joke_eval_list]
                    ),
                    DefaultTopicId(type="joke_contest_summary", source=ctx.topic_id.source)
                )
            else:
                await self.publish_message(JokeGeneratorRequest(round=message.round+1),
                                           topic_id=DefaultTopicId(type="joke_generator", source="user"))
        except ValidationError as e:
            print(e)

finish_event = asyncio.Event()

class JokeContestCommentatorAgent(RoutedAgent):
    def __init__(self, message_queue: asyncio.Queue, model_client: ChatCompletionClient):
        super().__init__("Joke Contest Commentator Agent")
        self._message_queue = message_queue
        self._model_client = model_client
        self._system_message = SystemMessage(content="""
你是一個專業的笑話比賽評論員，你的任務是用笑話的形式, 總結這次的笑話比賽。

# Notes
- 總是用繁體中文回答。
- 只要輸出總結，不要有其他內容。
- 你可以吐槽、諷刺或稱讚參賽者的表現。
- 盡可能使用 emoji 來表達情感。
- 總結不要太長，否則會被認為是廢話。
""")

    @message_handler
    async def handle_joke_evaluation(self, message: JokeContestSummaryRequest, ctx: MessageContext) -> None:
        prompt = f"請根據以下笑話比賽記錄，以幽默方式總結比賽（你可以吐槽、諷刺或稱讚）並給出你的評論：\n\n"
        for i, (joke, score, reason) in enumerate(zip(message.jokes, message.joke_eval_score, message.joke_eval_reason), 1):
            prompt += f"第 {i} 輪：\n笑話：{joke}\n分數：{score}\n理由：{reason}\n\n"
        prompt += "請給出你的總結評論："
        llm_result = await self._model_client.create(
            messages=[self._system_message, UserMessage(content=prompt, source=self.id.key)],
            cancellation_token=ctx.cancellation_token,
            json_output=JokeContestConclusion
        )
        content = llm_result.content
        try:
            conclusion = JokeContestConclusion.model_validate_json(content)
            print("-" * 20)
            print(f"比賽總結: {conclusion.summary}")

            finish_event.set()
            await self._message_queue.put(conclusion)

        except ValidationError as e:
            print(e)

async def run_autogen_chat():
    global joke_list
    global joke_eval_list
    joke_list = []
    joke_eval_list = []

    runtime = SingleThreadedAgentRuntime()
    model_client = OllamaChatCompletionClient(
        model="gemma2:latest",  # 使用最新的 Gemma 模型
    )

    message_queue = asyncio.Queue()
    runtime.start()

    await JokeGeneratorAgent.register(runtime, "joke_generator", lambda: JokeGeneratorAgent(message_queue, model_client))
    await runtime.add_subscription(DefaultSubscription(topic_type="joke_generator", agent_type="joke_generator"))

    await JokeEvaluationAgent.register(runtime, "joke_evaluation", lambda: JokeEvaluationAgent(message_queue, model_client, max_rounds=3))
    await runtime.add_subscription(DefaultSubscription(topic_type="joke_evaluation", agent_type="joke_evaluation"))

    await JokeContestCommentatorAgent.register(runtime, "joke_contest_summary", lambda: JokeContestCommentatorAgent(message_queue, model_client))
    await runtime.add_subscription(DefaultSubscription(topic_type="joke_contest_summary", agent_type="joke_contest_summary"))

    await runtime.publish_message(JokeGeneratorRequest(round=1), topic_id=DefaultTopicId(type="joke_generator", source="user"))

    while not finish_event.is_set():
        msg = await message_queue.get()
        yield msg

    finish_event.clear()
    await runtime.close()

async def on_start(history):
    history.clear()
    yield gr.update(interactive=False), history, gr.update(value="")

    final_summary = ""
    async for message in run_autogen_chat():
        if isinstance(message, JokeContestConclusion):
            final_summary = message.summary
        else:
            history.append(message)
        yield gr.update(interactive=False), history, gr.update(value="")

    yield gr.update(interactive=True), history, gr.update(value=final_summary)

with gr.Blocks() as demo:
    with gr.Column():
        start_btn = gr.Button("開始你們的表演")
        chatbot = gr.Chatbot(type="messages", height=800, label="笑話生成與評分")
        summary = gr.Textbox(placeholder="這裡會顯示笑話比賽的總結評論", label="笑話比賽總結", lines=3, interactive=False)

    start_btn.click(on_start, [chatbot], [start_btn, chatbot, summary])

if __name__ == "__main__":
    demo.queue().launch()