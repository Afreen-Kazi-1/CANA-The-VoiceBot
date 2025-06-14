

def main():
    context = []
    while(1):
        user_input, language = asr_model() #transcribe
        context.append("User Input: " + user_input)
        sa = sentiment_analysis(user_input, language) #sentiment
        data = rag_model(user_input, language) #chatbot
        system_response = middleman(user_input, context, data, sa) #middleman
        context.append("System Response: " + system_response)
        print(system_response)


