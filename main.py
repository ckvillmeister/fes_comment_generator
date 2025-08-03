from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-medium")
model = TFGPT2LMHeadModel.from_pretrained("gpt2-medium")

# Prompt input
input_text = """
Example:
Criteria:
- Lesson Preparation: 5
- Subject Mastery: 4
- Clarity of Instruction: 5
- Classroom Management: 3
- Use of Teaching Aids: 5

Comment:
The faculty shows excellent preparation and clarity in instruction. Subject mastery and use of teaching aids are commendable. However, classroom management can be improved.

Now generate a comment for the following:

Criteria:
- Lesson Preparation: 5
- Subject Mastery: 4
- Clarity of Instruction: 5
- Classroom Management: 2
- Use of Teaching Aids: 3

Comment:
"""

input_ids = tokenizer.encode(input_text, return_tensors='tf')

# Generate continuation
output = model.generate(
    input_ids,
    max_length=512,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    temperature=0.7,
    eos_token_id=tokenizer.eos_token_id
)

# Decode and print output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
