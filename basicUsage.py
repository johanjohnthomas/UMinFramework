from transformers import AutoModelForCausalLM, AutoTokenizer
print("Importing necessary libraries...")
from luh import AutoUncertaintyHead, CausalLMWithUncertainty


model_name = "mistralai/Mistral-7B-Instruct-v0.2"
uhead_name = "llm-uncertainty-head/uhead_Mistral-7B-Instruct-v0.2"
print("Reached here")
llm = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="cuda")
print("llm initialized")
tokenizer = AutoTokenizer.from_pretrained(
    model_name)
print("tokenizer initialized")
tokenizer.pad_token = tokenizer.eos_token
uhead = AutoUncertaintyHead.from_pretrained(
    uhead_name, base_model=llm)
llm_adapter = CausalLMWithUncertainty(llm, uhead, tokenizer=tokenizer)


messages = [
    [
        {
            "role": "user", 
            "content": "How many fingers are on a coala's foot?"
        }
    ]
]

chat_messages = [tokenizer.apply_chat_template(m, tokenize=False, add_bos_token=False) for m in messages]
inputs = tokenizer(chat_messages, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False).to("cuda")

output = llm_adapter.generate(inputs)
output["uncertainty_logits"]