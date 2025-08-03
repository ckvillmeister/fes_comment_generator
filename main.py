from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained("gpt2-medium")

# Prompt input
input_text = "The faculty member consistently demonstrates excellence in"
input_ids = tokenizer.encode(input_text, return_tensors='tf')

# Generate continuation
output = model.generate(
    input_ids,
    max_length=50,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)

# Decode and print output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
