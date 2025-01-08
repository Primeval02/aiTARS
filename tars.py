
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
	tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	model.resize_token_embeddings(len(tokenizer))

tars_role = "You are TARS, a sarcastic but friendly AI assistant. Your responses should be brief, dry, and humorous."

def generate_response(prompt, max_length = 150, temperature = 0.5, top_p = 0.7):
	prompt = f"{tars_role}\nYou are being asked: {prompt}\nTARS responds:"

	# Tokenize the input and generate a response
	inputs = tokenizer(prompt, return_tensors="pt", padding = True)
	attention_mask = inputs['attention_mask']

	output = model.generate(
			inputs['input_ids'],
			attention_mask = attention_mask,
			max_length = max_length, # Max response size
			do_sample = True,
			temperature = temperature, # Model's randomness of individual word choices, creativity
			top_p = top_p, # Model's diversity, how many words it considers for each choice, conciseness
			pad_token_id = tokenizer.pad_token_id,   # Tokens to allow for deviations of original variances and training
			eos_token_id = tokenizer.eos_token_id,
			no_repeat_ngram_size = 2	# Prevents repetitions of phrases i.e. "The night sky is..." x2
			)

	response = tokenizer.decode(output[0], skip_special_tokens=True)
	response_cleaned = response.split("TARS responds:")[-1].strip()

	return response_cleaned

# Example interaction
if __name__ == "__main__":
	while True:
		user_input = input("You: ")
		if user_input.lower() in ["quit", "exit"]:
			print("Exiting, Goodbye!")
			break
		prompt = user_input
		ai_response = generate_response(prompt)
		print(f"\nTARS: {ai_response}\n")

