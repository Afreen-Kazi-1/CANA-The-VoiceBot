

def main():
    context = []
    while(1):
        user_input, language = asr_model()
        context.append("User Input: " + user_input)
        sa = sentiment_analysis(user_input, language)
        data = rag_model(user_input, language)
        system_response = middleman(user_input, context, data, sa)
        context.append("System Response: " + system_response)
        print(system_response)


