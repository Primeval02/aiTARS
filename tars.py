# My first aiTARS! -Bradley Tate
# To Do: Math library to perform math functions (maybe put in seperate file?), fix output A: artifact, phrases to adjust sarcasm, creativity

from transformers import AutoModelForCausalLM, AutoTokenizer
import math
import random

# Load the GPT-Neo model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
	tokenizer.add_special_tokens({'pad_token': '[PAD]'})
	model.resize_token_embeddings(len(tokenizer))

# ADD SYNONYMS i.e TIMES/OVER for * and /
# MATH SECTION
math_database = {
		'square root': lambda x: math.sqrt(x),
		'cube root': lambda x: x ** (1/3),
		'plus': lambda x, y: x + y,
		'minus': lambda x, y: x - y,
		'times': lambda x, y: x * y,
		'over' : lambda x, y: x / y,
		'power': lambda x, y: x ** y
		}

def is_math_query(query):
	math_keywords = ['square root', 'cube root', 'plus', 'minus', 'times', 'over', 'power']
	return any(keyword in query.lower() for keyword in math_keywords)

def handle_math_query(query):
	query = query.lower()
	for keyword, operation in math_database.items():
		if keyword in query:
			try:
				numbers = [float(word) for word in query.split() if word.replace('.', '', 1).isdigit()]
				if len(numbers) == 1:
					result = operation(numbers[0])
				elif len(numbers) == 2:
					result = operation(numbers[0], numbers[1])
				else:
					return "Please provide exactly two numbers for binary operations."
				return f"The result of {keyword} is {result}."
			except Exception as e:
				return f"Error processing math query: {e}"
	return "Sorry, I didn't understand the math query."

# Sarcasm Section
sarcastic_count = 0
max_sarcastic_responses = 2
sarcasm_chance = 0.6

def should_be_sarcastic():
	global sarcastic_count
	if sarcastic_count >= max_sarcastic_responses:
		sarcastic_count = 0
		return False
	is_sarcastic = random.random() < sarcasm_chance
	if is_sarcastic:
		sarcastic_count += 1
	else:
		sarcastic_count = 0
	return is_sarcastic

# GENERATES RESPONSE
# default values for TARS is 0.4 temperature and 0.6 top_p
def generate_response(prompt, max_length = 150, temperature = 0.4, top_p = 0.6):
	if is_math_query(prompt):
		return handle_math_query(prompt)
	else:
		if should_be_sarcastic():
			prompt = f"You are TARS, a sarcastic but friendly AI assistant. Your response should be brief, dry, and humorous.\n Avoid unnecessary detail, and no more than a three sentences.\nUser asks: {prompt}\nTARS:"
		else:
			prompt = f"You are TARS, a friendly AI assistant. Your response should be serious, helpful, and concise.\n Avoid unnecessary detail, and no more than three sentences.\nUser asks: {prompt}\nTARS:"

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
		response_cleaned = response.split("TARS:")[-1].strip()

		return response_cleaned
		#uncomment to see prompt and if sarcastic
		#return response 

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

