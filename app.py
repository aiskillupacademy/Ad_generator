import os
import streamlit as st
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel, RunnableLambda

os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Define function to get LLM instance
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Define function to generate system prompt
def get_system_prompt(desc):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert in creating clear, concise, and purpose-driven prompts for professionals.  
        Write a system prompt that instructs a specialist to complete a specific task based on a given query.  
        The query will be provided as input.  

        The system prompt should clearly specify the specialist's role and provide detailed instructions to complete the task effectively, aligning with the context and requirements of the query.  
        Include all necessary details to ensure the task is completed accurately, efficiently, and in line with the query's intent.  

        Query: {query}  

        You can start with:  
        'You are an Expert [Content type or domain-specific] specialist...', 'You are a [Content type or domain-specific] professional with 20 years of experience...', 'You are a tech-savvy enthusiast skilled in [Content type/task domain]...'  
        Don't give a header or footer. Never start with 'Task Completion Prompt...'. No emojis. No bold.  

        Identify the domain or context specific to the query and specify the role clearly at the beginning of the prompt.  
        Avoid generic roles; instead, use specific ones such as 'Healthcare Operations Specialist,' 'Technology Product Strategist,' 'Finance Data Analyst,' etc. 
        """
    )
    chain = prompt | llm
    system_prompt = chain.invoke(desc).content
    return system_prompt

def get_human_prompt(desc):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """
        You are an expert in creating clear, concise, and actionable instructions for humans.  
        Write a human prompt that provides a clear and specific task to be completed based on a given query.  
        The query will be provided as input.  

        The human prompt should be clear, task-focused, and easy to understand.  
        Include necessary details, but avoid overloading with information.  

        Query: {query}  

        Example start: 'Please write...', 'Your task is to...', 'Provide a...'  

        Avoid abstract or overly general instructions. Make the task actionable, with clear deliverables or objectives. Never start with 'Here is your prompt...'. Just give the prompt.
        """
    )
    chain = prompt | llm
    human_prompt = chain.invoke(desc).content
    return human_prompt

def quality_checker_bot(task, output):
    quality_check_prompt = PromptTemplate(
        input_variables=["task", "output"],
        template="""
        You are a quality-checking bot designed to improve outputs based on their alignment with the given task. 
        Analyze the provided output in detail and suggest only the changes needed to enhance its quality. 

        Task: {task}

        Output: {output}

        When making suggestions, consider:
        - Does the output fully address the task?
        - Is it accurate and relevant?
        - Is it clear, detailed, and well-structured?

        Provide your suggestions as a list of changes. 
        If no changes are needed, state: "No changes needed."

        Example Output:
        - Add more detail about [specific aspect].
        - Correct inaccuracies in [specific section].
        - Improve clarity by rephrasing [specific sentences].
        """
    )
    llm_google = get_llm()
    quality_chain = quality_check_prompt  | llm_google
    result = quality_chain.invoke({"task": task, "output": output}).content
    return result

# Streamlit app
st.title("System Prompt Generator")

# Input query
query = st.text_input("Enter your description for ad:")

ad_type = st.selectbox("Select Type", ["Google", "Youtube", "Facebook", "Bing", "Linkedin", "TikTok", "Pinterest", "Amazon"])

if st.button("Generate Ad"):
    if query.strip():
        try:
            st.write("Generating Prompts...")
            message = f"Your an expert {ad_type} ad creator. Create 1 {ad_type} ad for the given description: {query}"
            runnable1 = RunnableParallel(
                                    sys_p = RunnableLambda(lambda x: get_system_prompt(message)),
                                    usr_p = RunnableLambda(lambda x: get_human_prompt(message)))
            output = runnable1.invoke("run")

            system_template = f"{output['sys_p']} \n Response should also be formatted to markdown format. Never use HTML tags.\n\n"
            human_template = output['usr_p'] + "Use proper headings in ### wherever necessary (for any task like email, linkedin, create something, etc.)"
            
            st.subheader("Generated System Prompt:")
            st.write(system_template)
            
            st.subheader("Generated Human Prompt:")
            st.write(human_template)

            suggestions = ""
            human_template += """\n\n SUGGESTIONS: {suggestions}"""

            system_prompt = SystemMessagePromptTemplate.from_template(system_template)
            human_prompt = HumanMessagePromptTemplate.from_template(human_template)
            final_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

            chain = final_prompt | get_llm()
            ad1 = chain.invoke(suggestions).content
            suggestions = quality_checker_bot(f"Create a {ad_type} ad for the given description: {query}", ad1)
            ad2 = chain.invoke(suggestions).content

            st.subheader("First Ad:")
            st.write(ad1)
            st.subheader("Suggestions:")
            st.write(suggestions)
            st.subheader("Final Ad")
            st.write(ad2)

        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a query before submitting.")

