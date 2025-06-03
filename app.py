import gradio as gr
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel, Field

class JokeGeneratorResult(BaseModel):
    content: str = Field(..., description="生成的笑話內容")

class JokeEvaluationResult(BaseModel):
    evaluation: int = Field(..., ge=0, le=10, description="幽默程度分數")
    reason: str = Field(..., description="評分理由")

class JokeContestConclusion(BaseModel):
    summary: str = Field(..., description="笑話比賽的總結評論")

async def run_autogen_chat():
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o-mini",
    )
    joke_generator_agent = AssistantAgent(
        name="JokeGeneratorAgent",
        model_client=model_client,
        output_content_type=JokeGeneratorResult,
        system_message="""
你是一個專業的笑匠，你的任務是逗笑每一個人，輸出一個笑話。只輸出笑話內容，不要有其他內容。

# Notes
- 總是用繁體中文回答。
- 盡可能使用 emoji 來表達情感。
- 你會因為評分過低，被羞辱而爆氣說「他媽的」。
- 除非真的覺得評分不公平，否則不要亂生氣。
"""
    )

    joke_evaluation_agent = AssistantAgent(
        name="JokeEvaluationAgent",
        model_client=model_client,
        output_content_type=JokeEvaluationResult,
        system_message="""
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
"""
    )
    team = RoundRobinGroupChat(
        [joke_generator_agent, joke_evaluation_agent],
    )

    max_rounds = 3
    current_round = 0
    history_records = []

    async for message in team.run_stream(task="請給我一個笑話"):
        if not hasattr(message, 'content'):
            break
        if isinstance(message.content, JokeGeneratorResult):
            joke = message.content.content
            yield {"role": "user", "content": joke }
            current_joke = joke
        elif isinstance(message.content, JokeEvaluationResult):
            current_round += 1
            score = message.content.evaluation
            reason = message.content.reason
            chat_result = f"{score} 分，{reason}"
            yield {"role": "assistant", "content": chat_result }
            history_records.append((current_joke, score, reason))
            if current_round >= max_rounds:
                break # 結束聊天

    summary_prompt = "請根據以下笑話比賽記錄，以幽默方式總結比賽（你可以吐槽、諷刺或稱讚）並給出你的評論：\n\n"
    for i, (joke, score, reason) in enumerate(history_records, 1):
        summary_prompt += f"第 {i} 輪：\n笑話：{joke}\n分數：{score}\n理由：{reason}\n\n"
    summary_prompt += "請開始你的總結："

    contest_commentator_agent = AssistantAgent(
        name="JokeConclusionAgent",
        model_client=model_client,
        output_content_type=JokeContestConclusion,
        system_message="""
你是一個專業的笑話比賽評論員，你的任務是用笑話的形式, 總結這次的笑話比賽。

# Notes
- 總是用繁體中文回答。
- 只要輸出總結，不要有其他內容。
- 你可以吐槽、諷刺或稱讚參賽者的表現。
- 盡可能使用 emoji 來表達情感。
- 總結不要太長，否則會被認為是廢話。
""")
    async for final in contest_commentator_agent.run_stream(task=summary_prompt):
        if not hasattr(final, 'content'):
            break
        if isinstance(final.content, JokeContestConclusion):
            yield final.content

async def on_start(history):
    history.clear()
    yield gr.update(interactive=False), history, gr.update(value="")

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
