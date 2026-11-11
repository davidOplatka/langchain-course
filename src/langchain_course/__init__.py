import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv()


def main() -> None:
    info = """
The United States of America (USA), also known as the United States (U.S.) or America, is a country primarily located in North America. It is a federal republic consisting of 50 states and a federal capital district, Washington, D.C. The 48 contiguous states border Canada to the north and Mexico to the south, with the semi-exclave of Alaska in the northwest and the archipelago of Hawaii in the Pacific Ocean. The United States also asserts sovereignty over five major island territories and various uninhabited islands in Oceania and the Caribbean.[k] It has the world's fourth-largest total area[l] and third-largest population, exceeding 341 million.[m]

Paleo-Indians migrated from North Asia to North America at least 15,000 years ago and formed various civilizations. Beginning in 1607, British colonization established the Thirteen Colonies. The 18th-century American Enlightenment spread values of republicanism and liberalism there, and clashes with the British Crown over taxation without parliamentary representation, among other denied rights, sparked the American Revolution. Independence from Great Britain was formally declared in the Lee Resolution of July 2, 1776, two days before the Declaration of Independence. Victory in the 1775–1783 Revolutionary War confirmed international recognition of U.S. sovereignty.

The U.S. expanded westward through the purchase, settlement, and conquest of European- and Indigenous-controlled territory. As more states joined the Union, a sectional division over slavery led 11 Southern states to declare secession and form the Confederate States of America, fighting the Union in the Civil War of 1861–1865. With the United States' victory and reunification, slavery was abolished. By 1900, the U.S. emerged as a great power, a status solidified after its involvement in World War I. Following Japan's attack on Pearl Harbor in 1941, it entered World War II on the Allied side. The war's aftermath left the U.S. and the Soviet Union as rival superpowers, vying for geopolitical dominance during the Cold War. The Soviet Union's collapse in 1991 left the U.S. as the world's sole superpower, although it has faced competition from China in the 2020s.

The U.S. federal government is a republic under the Constitution, which created a separation of powers among legislative, executive, and judicial branches. Executive authority is vested in a president, while Congress is a bicameral national legislature composed of the House of Representatives (a lower house based on population) and the Senate (an upper house based on equal representation for each state). The Supreme Court exercises judicial review, under which it may hold laws or governmental actions unconstitutional. Federalism grants substantial autonomy to the 50 states.

A developed country, the United States ranks high in economic competitiveness, innovation, and higher education. Its economy accounts for over a quarter of nominal global GDP and has been the world's largest since about 1890. The U.S. is the wealthiest country, with the highest disposable household income per capita among OECD members, though its wealth inequality is highly pronounced. American culture, shaped by centuries of immigration, is diverse and globally influential. The U.S. makes up nearly a third of global military spending and is widely considered to have the most powerful armed forces in the world. A member of numerous international organizations, it plays a dominant role in global political, cultural, economic, and military affairs.
"""

    summary_template = """
Given the provided context about a {entity}, please:
1. Create a short summary
2. Write two interesting facts about it (things that may not be as well known universally)
3. Answer the following question: What was the significance of {famous_person} in this {entity}'s history?

Instructions:
* Do NOT include any other follow up text or messages.
* Do NOT use any outside information beyond the provided context.
  If a task requests information outside of the provided context, refuse to provide an answer for that task.

<context>
{information}
</context>
"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information", "entity", "famous_person"],
        template=summary_template,
    )

    # llm = ChatOllama(model="gemma3:270m", temperature=0.1)
    # llm = ChatOllama(model="gemma3:4b", temperature=0.15)
    llm = ChatOllama(model="gpt-oss:latest", temperature=0.1)
    # llm = ChatAnthropic(
    #     model="claude-haiku-5-5",
    #     temperature=0.25,
    #     max_tokens=200
    # )

    # LangChain runnable change which passes each result from the previous item to the next item in the chain (separated by |)
    chain = summary_prompt_template | llm
    response = chain.invoke(
        input={
            "information": info,
            "entity": "country",
            "famous_person": "Abraham Lincoln",
        }
    )
    print(response.content)


if __name__ == "__main__":
    main()
