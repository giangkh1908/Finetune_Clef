"""Prompt format dùng chung cho train / generate, đảm bảo train và inference khớp nhau tuyệt đối."""

SYSTEM_PROMPT = (
    "You are an expert SQLite data analyst. Given a database schema and a question "
    "written in Vietnamese, write ONE SQLite query that answers the question. "
    "Use only tables and columns that exist in the schema. "
    "Return only the SQL query, without explanation or markdown."
)


def build_messages(schema: str, question: str) -> list[dict]:
    user = f"### Database schema\n{schema}\n\n### Câu hỏi\n{question}\n\n### SQL"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def render_prompt(tokenizer, schema: str, question: str) -> str:
    """Prompt đã áp chat template, kết thúc ở chỗ assistant bắt đầu trả lời (tắt thinking)."""
    return tokenizer.apply_chat_template(
        build_messages(schema, question),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def render_completion(tokenizer, schema: str, question: str, sql: str) -> str:
    """Phần assistant (SQL + token kết thúc lượt) đúng như chat template sinh ra."""
    prompt = render_prompt(tokenizer, schema, question)
    full = tokenizer.apply_chat_template(
        build_messages(schema, question) + [{"role": "assistant", "content": sql}],
        tokenize=False,
        enable_thinking=False,
    )
    if full.startswith(prompt):
        return full[len(prompt):]
    # Một số template viết lại lượt assistant cuối (vd. khối <think>), khi đó tự nối token kết thúc.
    end = "<|im_end|>" if "<|im_end|>" in full else tokenizer.eos_token
    return sql + end + "\n"


def clean_sql(text: str) -> str:
    """Lấy câu SQL từ output của model (bỏ markdown fence, khối think, dấu ; cuối)."""
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.split("```", 1)[0]
    text = " ".join(text.split())
    return text.rstrip(";").strip()
