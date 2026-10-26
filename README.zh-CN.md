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
- **暂定行**：灰色草稿里已经说完整的句子会立刻列进记录，标成“暂定”并显示为灰色，准确文字到达后被替换；只有没说完的半句留在下面的灰色行里。
- **连接词加粗**（`b` 开关）：but、because、so、therefore、if、for example 等在英文里加粗，长句的结构更容易看清（不含 and / or，它们太多会淹没其他词）。中文用常规字重，所以中文的连词（但是、因为、所以、如果……）也能以粗体突出。
- **专注模式**（`f`）：只显示最近 3 句，字号放大；点一句才显示它的中文（`c` 显示/隐藏全部）。少看屏幕、多听老师，完整记录留到课后。
- 也能处理录音文件：`./run.sh --file lecture.mp3`。

## 安装

需要 Python 3.10+ 和麦克风。在 Intel MacBook Pro（macOS）上测试过，Windows/Linux 未测试。

```bash
git clone https://github.com/SyzenShen/lecture-captions.git
cd lecture-captions
./setup.sh          # 虚拟环境、依赖、离线英中翻译模型（约 100 MB）；首次运行会自动下载 Whisper small.en（约 460 MB）
./run.sh example
```

第一次使用时 macOS 会请求终端的麦克风权限。

## 使用

每门课（或每节课）在 `prompts/` 里放一个提示词文件，需要的话再在 `glossary/` 里放一个术语表；文件名的第一个词就是课程名
（`prompts/ngs_full.txt`、`prompts/ngs_3_rnaseq.txt`、`glossary/ngs.txt` 都属于课程 `ngs`）。

```bash
./run.sh ngs 3      # 课程 ngs 的第 3 课（课号可以省略，省略时用 ngs_full）
./run.sh ngs        # 这门课自己的提示词 + 术语表
./run.sh            # 列出所有课程，直接回车 = 上次那门
```

课程名写开头几个字母就行（`./run.sh mol`）。网页会自动打开；档位跟着电源走（电池：`battery`；插电：`balanced`；在 `config.local.sh` 里写 `CAPTIONS_PROFILE=plugged` 或用 `--profile plugged` 可以改）。启动时只打印五行：课程、档位、网页地址、按键、笔记位置。`./run.sh --list-courses` 查看找到了哪些课程。再省一步可以加个别名：`alias cap='~/lecture-captions/run.sh'`。

| 按键（终端或网页都可用） | |
|---|---|
| `s` | 把到目前为止的内容保存为 Markdown（`notes/`，或 `CAPTIONS_NOTES_DIR`） |
| `c` | 隐藏 / 显示中文 |
| `p` | 暂停 / 继续（暂停时麦克风真的被关闭） |
| `b` | 连接词加粗 开 / 关 |
| `f` | 专注模式：只显示最近 3 句，点击才显示中文（此时 `c` 显示/隐藏全部中文） |

常用选项：`--no-web`（只用终端，默认会打开网页）、`--prompt-file`（不用课程名时）（主题提示词，**只能写英文**，建议 500 字符以内）、
`--profile battery|balanced|plugged`（用电情况，见下一节，默认 balanced）、`--glossary`（课程术语表，见下文）、`--no-draft`（不要灰色草稿：电池模式下 CPU 少约 40%，但前几个词出现得更晚）、`--chunk`（每次识别的最长秒数，默认 6）、`--translator none`（只要英文）、
`--device N` / `--list-devices`（选择输入设备）。复制 `config.example.sh` 为 `config.local.sh`（不会被提交）可以设置笔记保存位置。

### 怎么写提示词

提示词会作为开场上下文交给 Whisper：写上课程名和 20–40 个你预计会出现的术语、缩写、函数名，用逗号分隔（见 `prompts/example_biology.txt`）。
在我们自己的测试里这是**提升最大的一项**：同一段真实课堂录音，用对题的提示词能听对 15 个关键术语中的 10 个，用别的课的提示词只有 6 个。
用错科目的提示词会让 Whisper 把提示里的词塞进字幕，所以换课就要换提示词。

## 用电档位

在一段 7 分钟的真实课堂录音上测的（老师语速快、有口音；2019 款 Intel MacBook Pro，6 核），整个程序实时运行，不含翻译；一个核 = 100%。准确率是与 Whisper large-v3 整段解码结果的词级差异（它与 turbo 之间本身有约 13% 的差异，即噪声下限）。“之前”指 2026-10-24 上午的版本，用同样的方法测。

| 档位 | 运行内容 | 之前 CPU | 现在 CPU | 准确率 之前 / 现在 | 首词出现 | 准确文字替换草稿 |
|---|---|---|---|---|---|---|
| `battery` | `small.en` 解码合并的约 6 秒片段，beam 3，1 线程，无草稿 | 122% | **52%** | 25.7% / 23.6% | - | 约 12 秒后 |
| `balanced`（默认） | + `tiny.en` 灰色草稿 | 179% | **87%** | 25.7% / 18.9% | 中位数 1.6 秒（p90 3.5 秒） | 中位数 12.7 秒（p90 18 秒） |
| `plugged` | + `large-v3-turbo` 每约 15–27 秒（在停顿处结束）重读一个片段并替换其句子 | 约 410%（尚未用上述节省重测） | - | 17.8–21.6%（3 段，节省之前测的） | 同 `balanced` | 片段结束后约 10 秒 |

```bash
./run.sh --profile plugged my_course
```

省电的来源：语音按“到一个停顿处”合并后一次解码（`--merge-seconds`），而不是每个约 3.5 秒的片段单独解码（每次解码都会把音频补到 15 秒，短片段浪费大部分计算，更长的上下文也读得更准）；OpenMP 线程被动等待而不是空转；在这么长的片段上 beam 3 和单线程不影响准确率；草稿在听到 1 秒语音后就出第一版，之后每 1.5 秒更新。代价是准确文字比以前（6 秒）更晚到；`--merge-seconds 0` 可以恢复旧的节奏，CPU 约翻倍。中文跟随准确文字。

`--refine-model`、`--refine-threads`、`--block-seconds`、`--beam` 可以单独设置。精校落后超过 3 个片段时，最旧的片段保持原样。第一次运行 `plugged` 会下载 `large-v3-turbo`（约 1.6 GB）。

灰色草稿本来就很粗糙：约 40–47% 的词和准确文字不同，请把它当作“现在正在说什么”，而不是笔记。用 turbo 重读片段必须带温度回退（程序已经这样做）；不带的话，在一节课上 turbo 会陷入重复，差异是 35% 而不是 19.5%。

## 术语表：减少有误导性的翻译

离线翻译不懂专业词，而且一次只看一句：“galvanometric mirror system”被译成“相机镜像系统”，“calcium indicator lines”被译成“钙指标线”。而 Whisper 听错时（把 multi-well plates 听成 multi-world plates），翻译会把错误变成很通顺的中文。给一门课写一个术语表可以同时解决这两件事：

```
calcium indicator line = 钙离子指示剂品系     # 这个术语按我写的翻译
~ multi-world plate = multi-well plate      # Whisper 总是听错：直接修正英文
```

```bash
./run.sh my_course                  # 自动用 glossary/my_*.txt；也可以 --glossary 文件（可以给多个）
```

翻译前先把术语换成占位符，翻译后再换回，并显示成你听到的英文加括号里的中文，在网页上是深红色：multi-well plates（多孔板）。英文就是老师说的词，所以即使翻译器猜错也不会误导你（`--brackets first`：只有第一次出现时带中文，之后只显示中文）。不好翻译的术语可以写成 `galvanometric mirror = -`，这样这句话里就只显示英文。如果翻译器丢掉了占位符，这一句就不用术语表重新翻译。
术语表文件一改动就会重新读取，所以课间加一个术语，下一句就生效，不用重启。名为 `glossary/common_*.txt` 的术语表对所有课程生效（比如基础生化）。示例见 `glossary/example_biology.txt`；你自己放在 `glossary/` 里的术语表不会被提交。更大的离线模型没有帮助：NLLB-200（600M 和 1.3B）会丢字、长句翻到一半就停，CPU 是现在翻译器的 7–12 倍。

## 课后：生成准确的笔记

```bash
./refine.py transcripts/2026-10-23_1051.wav --prompt-file prompts/my_course.txt
```

用更强的模型（默认 `large-v3-turbo`，需下载约 1.6GB）把保存下来的整段录音一次性重新转写，生成 `<名字>.refined.md`，带时间、中文和提问清单。在一段真实的 7 分钟课堂录音上，英文离 Whisper large-v3 参照的距离是 13.4%（和整段 turbo 一样，已经在这个比较的噪声底线上：turbo 与 large-v3 彼此就差 12.9%），而实时输出是 25.7%。耗时约为录音长度的 0.8 倍，请在充电时运行。音质差时大模型帮助有限：同一段录音上 `medium.en` 并不比 `small.en` 好。

## 局限（如实说明）

- **只支持英文课**（`small.en`）。模型小是因为要在笔记本 CPU 上实时运行。
- **准确率受口音、距离和噪声影响很大**，这里没有正式评测。带噪声的合成语音上词错误率约 13–18%；
  真实教室里有口音的老师和学生提问更难，专业术语错得最多。
- 离线中文翻译比较基础，有时是直译（"transcription"→抄写），而且看不到上下文。
  `--translator claude`（需要 `pip install anthropic` 和 `ANTHROPIC_API_KEY`）更好，但**会把英文文本发给 API**。
- 延迟（在一段真实的 7 分钟课堂录音上测的：老师语速快、有口音，2019 款 Intel MacBook Pro）：一段话的前几个词会以灰色草稿的形式在说出后**约 1.6 秒**（中位数，p90 3.5 秒）出现，之后约 12.7 秒（中位数，`balanced` 档）被准确的文字替换，取舍见“用电档位”。最初的版本要 9.9 秒才显示。
- 实时文字为了快，是一小段一小段听的，准确率比能达到的低；课后请用 `refine.py`。
- 程序提示和网页按钮目前是中文。

## 隐私与录音

音频只在你的电脑上处理，不会上传；网页只监听 `127.0.0.1`。只有 `--translator claude` 会向外发送内容（英文文本，不含音频）。
录音和记录保存在 `transcripts/`，已被 git 忽略。**录课前请先确认学校的规定**，也不要公开他人的录音。
