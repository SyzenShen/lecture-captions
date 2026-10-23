# lecture-captions（课堂实时字幕）

**完全在你自己的电脑上运行**的英文课堂实时字幕，并附带中文翻译。
用 Whisper（[faster-whisper](https://github.com/SYSTRAN/faster-whisper)，CPU、int8）听写，
离线模型（[Argos Translate](https://github.com/argosopentech/argos-translate)）翻译，再用一个本地网页显示。
作者是坐在第一排、听英文课吃力的学生，用的是 MacBook。

[English](README.md)

## 功能

- 听麦克风，每次显示**一整句**（英文 + 中文），没说完的半句以灰色显示。
- **本地网页**，一键操作：暂停（真正关闭麦克风，几乎 0% CPU）、隐藏/显示中文（排版不跳动）、保存为 Markdown、调字号。
- **主题提示词**（`--prompt-file`）：给一串这节课的词汇，让 Whisper 更倾向写对专业词。
- 用 **Silero 语音检测**判断停顿，安静的录音和嘈杂的教室都能用（纯音量门限不行）。
- 保存本次的录音和逐句记录（`--no-record` 可关闭）；按 `s` 导出整节课的 Markdown。
- **提问会被高亮**，**最近 3 个问题置顶**在页面顶部：老师抛出问题、教室开始讨论时，瞥一眼就知道在聊什么。Markdown 笔记末尾会列出这节课所有的提问。
- **专注模式**（`f`）：只显示最近 3 句，字号放大；点一句才显示它的中文（`c` 显示/隐藏全部）。少看屏幕、多听老师，完整记录留到课后。
- 也能处理录音文件：`./run.sh --file lecture.mp3`。

## 安装

需要 Python 3.10+ 和麦克风。在 Intel MacBook Pro（macOS）上测试过，Windows/Linux 未测试。

```bash
git clone https://github.com/SyzenShen/lecture-captions.git
cd lecture-captions
./setup.sh          # 虚拟环境、依赖、离线英中翻译模型（约 100 MB）；首次运行会自动下载 Whisper small.en（约 460 MB）
./run.sh --web --prompt-file prompts/example_biology.txt
```

第一次使用时 macOS 会请求终端的麦克风权限。

## 使用

| 按键（终端或网页都可用） | |
|---|---|
| `s` | 把到目前为止的内容保存为 Markdown（`notes/`，或 `CAPTIONS_NOTES_DIR`） |
| `c` | 隐藏 / 显示中文 |
| `p` | 暂停 / 继续（暂停时麦克风真的被关闭） |
| `f` | 专注模式：只显示最近 3 句，点击才显示中文（此时 `c` 显示/隐藏全部中文） |

常用选项：`--web`（网页）、`--prompt-file`（主题提示词，**只能写英文**，建议 500 字符以内）、
`--chunk 5`（更低延迟；默认 8 秒，在我们的测试里 CPU 少约 29%、准确率相当）、`--translator none`（只要英文）、
`--device N` / `--list-devices`（选择输入设备）。复制 `config.example.sh` 为 `config.local.sh`（不会被提交）可以设置笔记保存位置。

### 怎么写提示词

提示词会作为开场上下文交给 Whisper：写上课程名和 20–40 个你预计会出现的术语、缩写、函数名，用逗号分隔（见 `prompts/example_biology.txt`）。
在我们自己的测试里这是**提升最大的一项**：同一段真实课堂录音，用对题的提示词能听对 15 个关键术语中的 10 个，用别的课的提示词只有 6 个。
用错科目的提示词会让 Whisper 把提示里的词塞进字幕，所以换课就要换提示词。

## 局限（如实说明）

- **只支持英文课**（`small.en`）。模型小是因为要在笔记本 CPU 上实时运行。
- **准确率受口音、距离和噪声影响很大**，这里没有正式评测。带噪声的合成语音上词错误率约 13–18%；
  真实教室里有口音的老师和学生提问更难，专业术语错得最多。
- 离线中文翻译比较基础，有时是直译（"transcription"→抄写），而且看不到上下文。
  `--translator claude`（需要 `pip install anthropic` 和 `ANTHROPIC_API_KEY`）更好，但**会把英文文本发给 API**。
- 字幕比说话人慢几秒。
- 程序提示和网页按钮目前是中文。

## 隐私与录音

音频只在你的电脑上处理，不会上传；网页只监听 `127.0.0.1`。只有 `--translator claude` 会向外发送内容（英文文本，不含音频）。
录音和记录保存在 `transcripts/`，已被 git 忽略。**录课前请先确认学校的规定**，也不要公开他人的录音。
