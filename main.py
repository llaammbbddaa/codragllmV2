# main.py
# november 11 2025

from openai import OpenAI

# dataset, not yet vector db
# just like the lines from the txt file
dataset = []
with open("courses.txt") as file:
	dataset = file.readlines()
	# print("loaded dataset")
	# print(str(len(dataset)) + " lines worth")

# openrouter init
client = OpenAI(
	base_url="https://openrouter.ai/api/v1",
	api_key="use your own",
)

# another benefit of python being dynamically typed, no need for method overloading
# the datatypes take care of themselves
def embedsInput(query):
	embedding = client.embeddings.create(
		model="qwen/qwen3-embedding-8b",
		input = query,
		encoding_format="float"
	)

	# pair each input string with its corresponding embedding vector
	results = [
		(text, item.embedding)
		for text, item in zip(dataset, embedding.data)
	]
	return results

# vector db generated from embedding model
# from tutorial
#       Each element in the VECTOR_DB will be a tuple (chunk, embedding)
#       The embedding is a list of floats, for example: [0.1, 0.04, -0.34, 0.21, ...]
vectorDB = embedsInput(dataset)

def cosine_similarity(a, b):
  dot_product = sum([x * y for x, y in zip(a, b)])
  norm_a = sum([x ** 2 for x in a]) ** 0.5
  norm_b = sum([x ** 2 for x in b]) ** 0.5
  return dot_product / (norm_a * norm_b)

def retrieve(query, top_n=3):
	query_embedding = embedsInput(query)[0][1]
	# temporary list to store (chunk, similarity) pairs
	similarities = []
	for chunk, embedding in vectorDB:
		similarity = cosine_similarity(query_embedding, embedding)
		similarities.append((chunk, similarity))
	# sort by similarity in descending order, because higher similarity>
	similarities.sort(key=lambda x: x[1], reverse=True)
	# finally, return the top N most relevant chunks
	return similarities[:top_n]


# input string for llm output
def inputMessage(userInput, client=client):
	completion = client.chat.completions.create(
		model="openai/gpt-4.1-mini",
		messages=[
			{
				"role": "user",
				"content": userInput
			}
		]
	)

	return completion.choices[0].message.content

previousMessages = ""
while True:
	query = input("enter request >> ")
	userInput = "user states >> " + query
	previousMessages = previousMessages + "\n" + userInput
	context = "use this information to answer the question >> \n" + str(retrieve(query))
	previousMessages = previousMessages + "\n" + context
	output = inputMessage(previousMessages)
	print(output)
	previousMessages = previousMessages + "\n" + "model states >> " + output

