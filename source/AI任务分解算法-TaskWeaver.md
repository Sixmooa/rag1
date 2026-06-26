# AI 任务分解算法 - Microsoft TaskWeaver

> **来源**: [Microsoft TaskWeaver](https://github.com/microsoft/TaskWeaver) (6.2k Stars, MIT License)
> **论文**: *TaskWeaver: A Code-First Agent Framework* (arXiv:2311.17541)
> **提取时间**: 2026-04-27

---

## 一、算法概述

TaskWeaver 是微软研究院开发的 **"代码优先"（code-first）智能体框架**，核心能力是将复杂的 AI 任务分解为可执行的子任务并协调多个插件完成。

### 核心特性

| 特性 | 描述 |
|------|------|
| **任务规划与分解** | 将复杂任务分解为子任务，并跟踪执行进度 |
| **反思执行** | 支持对执行过程进行反思和调整 |
| **丰富数据结构** | 支持在 Python 中操作 DataFrame 等复杂数据结构 |
| **自定义算法** | 可封装自己的算法为插件并进行编排 |
| **领域知识融合** | 轻松融合领域特定知识以提高可靠性 |
| **有状态执行** | 支持有状态的代码生成与执行 |
| **代码验证** | 执行前验证生成的代码，检测潜在问题 |

---

## 二、架构设计

```
用户请求
   |
   v
┌──────────┐
│ Planner  │  ← 任务分解核心（LLM 驱动）
│  规划器   │
└────┬─────┘
     │ 分解后的子任务
     v
┌──────────┐     ┌──────────┐     ┌──────────┐
│ Worker 1 │     │ Worker 2 │     │ Worker N │
│ 插件执行  │ ... │ 代码生成  │ ... │ 数据分析  │
└──────────┘     └──────────┘     └──────────┘
     │               │               │
     v               v               v
┌─────────────────────────────────────────┐
│              Memory / 共享内存            │
│    (聊天历史 + 代码执行历史 + 内存数据)    │
└─────────────────────────────────────────┘
```

### 任务分解流程

1. **接收请求** → Planner 接收用户自然语言请求
2. **任务分解** → LLM 将任务分解为多个子任务（init_plan）
3. **生成计划** → 按步骤生成执行计划（plan）
4. **分派执行** → 将子任务分配给对应 Worker
5. **结果汇总** → 收集各子任务结果并生成最终响应

---

## 三、核心脚本：Planner 任务分解算法

### 3.1 Planner 主类（planner.py）

来源文件: `taskweaver/planner/planner.py`

```python
import datetime
import json
import os
import types
from json import JSONDecodeError
from typing import Dict, Iterable, List, Optional

from injector import inject
from taskweaver.llm import LLMApi
from taskweaver.llm.util import ChatMessageType, format_chat_message
from taskweaver.logging import TelemetryLogger
from taskweaver.memory import Memory, Post, Round, RoundCompressor
from taskweaver.memory.attachment import AttachmentType
from taskweaver.memory.experience import ExperienceGenerator
from taskweaver.memory.memory import SharedMemoryEntry
from taskweaver.module.event_emitter import SessionEventEmitter
from taskweaver.module.tracing import Tracing, tracing_decorator
from taskweaver.role import PostTranslator, Role
from taskweaver.role.role import RoleConfig
from taskweaver.utils import read_yaml


class PlannerConfig(RoleConfig):
    def _configure(self) -> None:
        self._set_name("planner")
        self.prompt_file_path = self._get_path(
            "prompt_file_path",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "planner_prompt.yaml",
            ),
        )
        self.prompt_compression = self._get_bool("prompt_compression", False)
        self.compression_prompt_path = self._get_path(
            "compression_prompt_path",
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "compression_prompt.yaml",
            ),
        )
        self.llm_alias = self._get_str("llm_alias", default="", required=False)


class Planner(Role):
    conversation_delimiter_message: str = "Let's start the new conversation!"

    @inject
    def __init__(
        self,
        config: PlannerConfig,
        logger: TelemetryLogger,
        tracing: Tracing,
        event_emitter: SessionEventEmitter,
        llm_api: LLMApi,
        workers: Dict[str, Role],
        round_compressor: Optional[RoundCompressor],
        post_translator: PostTranslator,
        experience_generator: Optional[ExperienceGenerator] = None,
    ):
        super().__init__(config, logger, tracing, event_emitter)
        self.config = config
        self.alias = "Planner"
        self.llm_api = llm_api
        self.workers = workers
        self.recipient_alias_set = set([alias for alias, _ in self.workers.items()])
        self.planner_post_translator = post_translator
        self.prompt_data = read_yaml(self.config.prompt_file_path)
        self.instruction_template = self.prompt_data["instruction_template"]
        self.response_json_schema = json.loads(
            self.prompt_data["response_json_schema"]
        )
        # restrict the send_to field to the recipient alias set
        self.response_json_schema["properties"]["response"]["properties"][
            "send_to"
        ]["enum"] = list(self.recipient_alias_set) + ["User"]
        self.ask_self_cnt = 0
        self.max_self_ask_num = 3
        self.round_compressor = round_compressor
        self.compression_prompt_template = read_yaml(
            self.config.compression_prompt_path
        )["content"]
        self.experience_generator = experience_generator
        self.experience_loaded_from = None
        self.logger.info("Planner initialized successfully")

    def compose_sys_prompt(self, context: str):
        worker_description = ""
        for alias, role in self.workers.items():
            worker_description += (
                f"###{alias}\n"
                f"- The name of this Worker is `{alias}`\n"
                f"{role.get_intro()}\n"
                f'- The message from {alias} will start with "From: {alias}"\n'
            )
        instruction = self.instruction_template.format(
            environment_context=context,
            response_json_schema=json.dumps(self.response_json_schema),
            worker_intro=worker_description,
        )
        return instruction

    def format_message(self, role: str, message: str) -> str:
        return f"From: {role}\nMessage: {message}\n"

    def compose_conversation_for_prompt(
        self,
        conv_rounds: List[Round],
        summary: Optional[str] = None,
    ) -> List[ChatMessageType]:
        conversation: List[ChatMessageType] = []
        for rnd_idx, chat_round in enumerate(conv_rounds):
            conv_init_message = None
            if rnd_idx == 0:
                conv_init_message = Planner.conversation_delimiter_message
            if summary is not None:
                self.logger.debug(f"Summary: {summary}")
                summary_message = (
                    f"\nThe context summary of the Planner's previous rounds"
                    f" can refer to:\n{summary}\n\n"
                )
                conv_init_message += "\n" + summary_message
            for post in chat_round.post_list:
                if post.send_from == self.alias:
                    if (
                        post.send_to == "User"
                        or post.send_to in self.recipient_alias_set
                    ):
                        # planner responses
                        planner_message = self.planner_post_translator.post_to_raw_text(
                            post=post,
                        )
                        conversation.append(
                            format_chat_message(
                                role="assistant",
                                message=planner_message,
                            ),
                        )
                    elif post.send_to == self.alias:
                        # self correction for planner response
                        conversation.append(
                            format_chat_message(
                                role="assistant",
                                message=post.get_attachment(
                                    type=AttachmentType.invalid_response,
                                )[0].content,
                            ),
                        )
                        conversation.append(
                            format_chat_message(
                                role="user",
                                message=self.format_message(
                                    role="User",
                                    message=post.get_attachment(
                                        type=AttachmentType.revise_message
                                    )[0].content,
                                ),
                            ),
                        )
                else:
                    # messages from user or workers
                    conversation.append(
                        format_chat_message(
                            role="user",
                            message=self.format_message(
                                role=post.send_from,
                                message=post.message
                                if conv_init_message is None
                                else conv_init_message + "\n" + post.message,
                            ),
                            image_urls=[
                                attachment.extra["image_url"]
                                for attachment in post.get_attachment(
                                    type=AttachmentType.image_url
                                )
                            ],
                        ),
                    )
                conv_init_message = None
        return conversation

    def get_env_context(self) -> str:
        now = datetime.datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")
        return f"- Current time: {current_time}"

    def compose_prompt(
        self,
        rounds: List[Round],
    ) -> List[ChatMessageType]:
        experiences = self.format_experience(
            template=self.prompt_data["experience_instruction"],
        )
        chat_history = [
            format_chat_message(
                role="system",
                message=f"{self.compose_sys_prompt(context=self.get_env_context())}"
                f"\n{experiences}",
            ),
        ]
        for conv_example in self.examples:
            conv_example_in_prompt = self.compose_conversation_for_prompt(
                conv_example.rounds,
            )
            chat_history += conv_example_in_prompt
        summary = None
        if self.config.prompt_compression and self.round_compressor is not None:
            summary, rounds = self.round_compressor.compress_rounds(
                rounds,
                rounds_formatter=lambda _rounds: str(
                    self.compose_conversation_for_prompt(_rounds),
                ),
                prompt_template=self.compression_prompt_template,
            )
        chat_history.extend(
            self.compose_conversation_for_prompt(
                rounds,
                summary=summary,
            ),
        )
        return chat_history

    @tracing_decorator
    def reply(
        self,
        memory: Memory,
        prompt_log_path: Optional[str] = None,
        **kwargs: ...,
    ) -> Post:
        rounds = memory.get_role_rounds(role=self.alias)
        assert len(rounds) != 0, "No chat rounds found for planner"

        user_query = rounds[-1].user_query
        self.tracing.set_span_attribute("user_query", user_query)
        self.tracing.set_span_attribute("use_experience", self.config.use_experience)

        self.role_load_experience(query=user_query, memory=memory)
        self.role_load_example(
            role_set=set(self.recipient_alias_set) | {self.alias, "User"},
            memory=memory,
        )

        post_proxy = self.event_emitter.create_post_proxy(self.alias)
        post_proxy.update_status("composing prompt")
        chat_history = self.compose_prompt(rounds)

        def check_post_validity(post: Post):
            missing_elements: List[str] = []
            validation_errors: List[str] = []
            if not post.send_to or post.send_to == "Unknown":
                missing_elements.append("send_to")
            if post.send_to == self.alias:
                validation_errors.append(
                    "The `send_to` field must not be `Planner` itself"
                )
            if not post.message or post.message.strip() == "":
                missing_elements.append("message")
            attachment_types = [attachment.type for attachment in post.attachment_list]
            if AttachmentType.init_plan not in attachment_types:
                missing_elements.append("init_plan")
            if AttachmentType.plan not in attachment_types:
                missing_elements.append("plan")
            if AttachmentType.current_plan_step not in attachment_types:
                missing_elements.append("current_plan_step")
            if len(missing_elements) > 0:
                validation_errors.append(
                    f"Missing elements: {', '.join(missing_elements)} in the `response` element"
                )
            assert len(validation_errors) == 0, ";".join(validation_errors)

        post_proxy.update_status("calling LLM endpoint")
        llm_stream = self.llm_api.chat_completion_stream(
            chat_history,
            use_smoother=True,
            llm_alias=self.config.llm_alias,
            json_schema=self.response_json_schema,
            stream=True,
        )

        llm_output: List[str] = []
        try:

            def stream_filter(s: Iterable[ChatMessageType]):
                is_first_chunk = True
                try:
                    for c in s:
                        if is_first_chunk:
                            post_proxy.update_status("receiving LLM response")
                            is_first_chunk = False
                        llm_output.append(c["content"])
                        yield c
                finally:
                    if isinstance(s, types.GeneratorType):
                        try:
                            s.close()
                        except GeneratorExit:
                            pass

            self.tracing.set_span_attribute(
                "prompt", json.dumps(chat_history, indent=2)
            )
            prompt_size = self.tracing.count_tokens(json.dumps(chat_history))
            self.tracing.set_span_attribute("prompt_size", prompt_size)
            self.tracing.add_prompt_size(
                size=prompt_size,
                labels={
                    "direction": "input",
                },
            )

            self.planner_post_translator.raw_text_to_post(
                post_proxy=post_proxy,
                llm_output=stream_filter(llm_stream),
                validation_func=check_post_validity,
            )

            plan = post_proxy.post.get_attachment(type=AttachmentType.plan)[0]
            bulletin_message = (
                f"\n====== Plan ======\n"
                f"I have drawn up a plan:\n{plan}\n"
                f"==================\n"
            )

            post_proxy.update_attachment(
                type=AttachmentType.shared_memory_entry,
                message="Add the plan to the shared memory",
                extra=SharedMemoryEntry.create(
                    type="plan",
                    scope="round",
                    content=bulletin_message,
                ),
            )

        except (JSONDecodeError, AssertionError) as e:
            self.logger.error(f"Failed to parse LLM output due to {str(e)}")
            self.tracing.set_span_status("ERROR", str(e))
            self.tracing.set_span_exception(e)

            post_proxy.error(f"Failed to parse LLM output due to {str(e)}")
            post_proxy.update_attachment(
                "".join(llm_output),
                AttachmentType.invalid_response,
            )
            post_proxy.update_attachment(
                f"Your JSON output has errors. {str(e)}. "
                "You must add or missing elements at in one go and send the response again.",
                AttachmentType.revise_message,
            )

            if self.ask_self_cnt > self.max_self_ask_num:
                self.ask_self_cnt = 0
                post_proxy.end(
                    f"Planner failed to generate response because {str(e)}"
                )
                raise Exception(
                    f"Planner failed to generate response because {str(e)}"
                )
            else:
                post_proxy.update_send_to(self.alias)
                self.ask_self_cnt += 1

        if prompt_log_path is not None:
            self.logger.dump_prompt_file(chat_history, prompt_log_path)

        reply_post = post_proxy.end()
        self.tracing.set_span_attribute("out.from", reply_post.send_from)
        self.tracing.set_span_attribute("out.to", reply_post.send_to)
        self.tracing.set_span_attribute("out.message", reply_post.message)
        self.tracing.set_span_attribute(
            "out.attachments", str(reply_post.attachment_list)
        )
        return reply_post
```

---

## 四、算法关键机制详解

### 4.1 任务分解（Task Decomposition）核心流程

```
用户请求 "从数据库拉取数据并进行异常检测"
   │
   ├── init_plan: [
   │     "1. 连接数据库并拉取相关数据",
   │     "2. 对数据进行预处理",
   │     "3. 应用异常检测算法",
   │     "4. 可视化并报告结果"
   │   ]
   │
   ├── plan: [
   │     "Step 1: 调用 sql_pull_data 插件获取数据 → 生成 DataFrame",
   │     "Step 2: 调用 data_preprocessing 插件清洗数据",
   │     "Step 3: 调用 anomaly_detection 插件检测异常",
   │     "Step 4: 调用 visualization 插件生成图表"
   │   ]
   │
   └── current_plan_step: 1
       └── send_to: "CodeGenerator" (将步骤1交给代码生成器执行)
```

### 4.2 自我纠正机制（Self-Correction）

Planner 内置了最多 3 次的自我纠正循环 (`max_self_ask_num = 3`)：

1. LLM 输出被验证（`check_post_validity`）
2. 如果 JSON 解析失败或缺少字段（`init_plan`, `plan`, `current_plan_step`）
3. 将错误信息附加为 `revise_message`，重新发给 Planner 自身
4. 超过最大重试次数则抛出异常

### 4.3 Prompt 压缩（Round Compression）

对于长对话历史，TaskWeaver 支持将历史轮次压缩为摘要：

```python
if self.config.prompt_compression and self.round_compressor is not None:
    summary, rounds = self.round_compressor.compress_rounds(
        rounds,
        rounds_formatter=lambda _rounds: str(
            self.compose_conversation_for_prompt(_rounds),
        ),
        prompt_template=self.compression_prompt_template,
    )
```

### 4.4 经验学习（Experience Learning）

Planner 可加载历史经验来指导任务分解：

```python
self.role_load_experience(query=user_query, memory=memory)
self.role_load_example(
    role_set=set(self.recipient_alias_set) | {self.alias, "User"},
    memory=memory,
)
```

---

## 五、相关算法对比

| 算法/框架 | 来源 | 分解策略 | 特点 |
|-----------|------|----------|------|
| **TaskWeaver** | Microsoft | LLM 驱动 + 代码生成 | 有状态执行、插件编排、自我纠正 |
| **HuggingGPT** | 浙大 | LLM 作为控制器 + 模型选择 | 多模态、Hugging Face 模型生态 |
| **Plan-and-Solve** | ACL 2023 | 先规划再执行的提示策略 | 零样本、无需手动示例 |
| **ReAct** | Yao et al. | 推理+行动交替 | 思考-行动-观察循环 |
| **Tree of Thoughts** | Princeton | 树状搜索多路径 | 探索多个推理路径 |

---

## 六、引用信息

```bibtex
@article{taskweaver,
  title={TaskWeaver: A Code-First Agent Framework},
  author={Bo Qiao, Liqun Li, Xu Zhang, Shilin He, Yu Kang, Chaoyun Zhang,
          Fangkai Yang, Hang Dong, Jue Zhang, Lu Wang, Minghua Ma,
          Pu Zhao, Si Qin, Xiaoting Qin, Chao Du, Yong Xu, Qingwei Lin,
          Saravan Rajmohan, Dongmei Zhang},
  journal={arXiv preprint arXiv:2311.17541},
  year={2023}
}
```

---

## 七、参考链接

- GitHub 仓库: https://github.com/microsoft/TaskWeaver
- 论文: https://arxiv.org/abs/2311.17541
- HuggingGPT 论文: https://arxiv.org/abs/2303.17580
- Plan-and-Solve 论文: https://arxiv.org/abs/2305.04091
