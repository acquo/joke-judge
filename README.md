---
title: Joke Judge
emoji: 📚
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.32.1
app_file: app.py
pinned: false
license: mit
---

# 🤡 笑話生成與評分大賽

這是一個使用 Gradio 與 OpenAI GPT-4o-mini 所打造的互動式 AI 應用，模擬一場「笑話生成與評分」的比賽。

由兩位 AI 擔綱演出：

- 🎭 **JokeGeneratorAgent**：負責創作笑話，力求讓你噴飯。
- 🎯 **JokeEvaluationAgent**：冷酷評審，毫不留情打分與吐槽。

當比賽完成後，還會由一位專業笑話評論員總結整場比賽內容！

---

## 🧠 技術架構

- **Gradio**：前端互動介面
- **Autogen Agents**：多智能體對話邏輯
- **OpenAI GPT-4o-mini**：語言模型核心
- **Pydantic**：輸出格式結構化驗證

---

## 🚀 如何使用

### 本地執行

1. 安裝套件（建議使用虛擬環境）

```bash
pip install -r requirements.txt