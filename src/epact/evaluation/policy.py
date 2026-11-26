"""Local model adapter for the frozen action-only policy."""

from epact.evaluation.protocol import parse_action, policy_prompt


def ask_model(model, processor, state, history, args, final: bool):
    import torch
    from PIL import Image
    import io as _io, base64 as _b64

    image = Image.open(_io.BytesIO(_b64.b64decode(state["frame_png_b64"]))).convert(
        "RGB"
    )
    text = policy_prompt(state, history, final, getattr(args, "hide_budget", False))
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to("cuda")
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
        )
    reply = processor.batch_decode(
        generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]
    parsed = parse_action(reply)
    args.last_model_reply = reply
    args.last_parse_failed = parsed is None
    if parsed is None:
        return ("Answer", {"answer": "no", "confidence": 0.0})
    return parsed
