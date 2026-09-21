from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()

def main():

    print("Hello, World!")

    information = """
    The world's number one richest person is Elon Musk.

    Wealth Details:
    Primary Sources: Major stakes in Tesla and SpaceX.
    Tracking: Forbes and the Bloomberg Billionaires Index track billionaire wealth.
    Fluctuation: Net worth amounts change daily based on global stock market
    activity and company valuations.
    """

    summary_template = """
    Give the following information about a person and create:

    1. A short summary
    2. Two interesting facts about them

    Information:
    {information}
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    '''llm = ChatOllama(
        model="llama3.1",
        temperature=0
    )'''

    llm=ChatOllama(
        model="gemma3:270m",
        temperature=0
    )

    summary_chain = summary_prompt_template | llm

    summary = summary_chain.invoke(input=
        {"information": information}
    )

    print(summary.content)


if __name__ == "__main__":
    main()

# from dotenv import load_dotenv
# import os

# load_dotenv()

# print("Tracing:", os.getenv("LANGSMITH_TRACING"))
# print("Endpoint:", os.getenv("LANGSMITH_ENDPOINT"))
# print("Project:", os.getenv("LANGSMITH_PROJECT"))
# print("API key loaded:", bool(os.getenv("LANGSMITH_API_KEY")))