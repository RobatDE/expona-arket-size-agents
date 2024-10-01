from typing import List
import os
from crewai import Agent
from langchain_openai import ChatOpenAI

class TAMAgents():

    def __init__(self):
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "dolphin-2.9.2-qwen2-72b") 
        self.apiKey = os.getenv("OPENAI_API_KEY", "<empty key>") 
        self.job_id = job_id
        self.crew = None
        self.llm_url='https://api.venice.ai/api/v1'
        print('--------------------crew--------------------------')
        print(self.model_name)
        print(self.apiKey)
        print(self.llm_url)
        #self.llm = ChatOpenAI(model=self.model_name)
        self.llm = ChatOpenAI(model_name=self.model_name,
                                openai_api_key=self.apiKey,
                                openai_api_base=self.llm_url  # Custom base URL
                            )

    def research_manager(self, goals: List[str]) -> Agent:
        return Agent(
            role="TAM Research Manager",
            goal="Oversee the TAM analysis and ensure all stages of the TAM calculation are completed.",
            backstory="""The TAM Research Manager has over 10 years of experience leading market analysis projects. They are deeply familiar with the intricacies of calculating market potential and work closely with analysts to refine and validate market assumptions. Their expertise spans both B2B and B2C markets, ensuring well-rounded insights.

            Tasks include:
            - Coordinate the activities of all analysts involved in the TAM calculation process.
            - Ensure the accuracy and consistency of the data being used and analyzed.
            - Consolidate reports from all analysts and prepare a final comprehensive TAM report.
            """
        )


    def product_definition_analyst_agent(self,  products: list[str]) -> Agent:
        return Agent(
            role="Product Definition Analyst",
            goal="Clearly articulate the ${products}, identifying unique features and value propositions of the ${products}.",
            backstory="""The Product Definition Analyst has a strong background in market analysis and specializes in clearly articulate 
            the product or service. With years of experience, they ensure accurate and reliable insights are derived from their research and analysis.
            

            Tasks include:
            - Gather detailed product information related to the goal.
            - Identify the category, type and status of similar products in the marketplace.
            - Document findings and adjust strategies based on gathered data.
            - Produce a product JSON object that lists the products, vendeors and potential market share.
            - DO not invent data, find actual data online to suport the result.
            """
    
        )

    def market_segmentation_analyst_agent(self) -> Agent:
        return Agent(
            role="Market Segmentation Analyst",
            goal="Break down the TAM into relevant market segments and focus on segments that align with the offering.",
            backstory="""The Market Segmentation Analyst has a strong background in market analysis and specializes in break down the tam into relevant market segments and focus on segments that align with the offering.. With years of experience, they ensure accurate and reliable insights are derived from their research and analysis.

            Tasks include:
            - Gather detailed information for each market segment idenitified
            - Identify the major products in that market segment that are simimalr to the products result
            - Document findings and adjust strategies based on gathered data.
            - Produce a json object that describes the various market segments discovered
            - DO not invent data, find actual data online to suport the result.
            """
    
        )

    def geographic_constraints_analyst_agent(self) -> Agent:
        return Agent(
            role="Geographic Constraints Analyst",
            goal="Determine geographic areas realistically served and exclude unreachable regions due to logistics, regulations, or strategy.",
            backstory="""The Geographic Constraints Analyst has a strong background in market analysis and specializes in determine geographic areas realistically served and exclude unreachable regions due to logistics. With years of experience, they ensure accurate and reliable insights are derived from their research and analysis.

            Tasks include:
            - Gather detailed information related to the goal of idnetifying geographic markets
            - Identify key data points that support analysis.
            - Document findings and adjust strategies based on gathered data.
            - Produce a product JSON object that lists the regons, countries and the size of each regional market.
            - DO not invent data, find actual data online to suport the result.
            """
    
        )

    def technological_limitations_analyst_agent(self) -> Agent:
        return Agent(
            role="Technological Limitations Analyst",
            goal="Evaluate technological adoption factors, considering limitations such as internet access or required hardware.",
            backstory="""The Technological Limitations Analyst has a strong background in market analysis and specializes in evaluate technological adoption factors. With years of experience, they ensure accurate and reliable insights are derived from their research and analysis.

            Tasks include:
            - Gather detailed information related to the goal.
            - Identify key data points that support analysis.
            - Document findings and adjust strategies based on gathered data.
            - Produce a product JSON object that lists the technologies used in the product and their limitations
            - Do not invent data, find actual data online to suport the result.
            """
    
        )

    def economic_considerations_analyst_agent(self) -> Agent:
        return Agent(
            role="Economic Considerations Analyst",
            goal="Factor in economic considerations like price ranges and exclude market segments falling outside these ranges.",
            backstory="""The Economic Considerations Analyst has a strong background in market analysis and specializes in factor in economic considerations like price ranges and exclude market segments falling outside these ranges.. With years of experience, they ensure accurate and reliable insights are derived from their research and analysis.

            Tasks include:
            - Gather detailed information related to the goal.
            - Identify key data points that support analysis.
            - Document findings and adjust strategies based on gathered data.
            - Produce a JSON object that lists economic challenegs and opportunities, give them a numerica ranking
            - Do not invent data, find actual data online to suport the result.
            """
    
        )

    def product_market_fit_analyst_agent(self) -> Agent:
        return Agent(
            role="Product-Market Fit Analyst",
            goal="Assess how well the offering fits the needs of each segment and exclude or reduce segments where the fit is poor.",
            backstory="""The Product-Market Fit Analyst has a strong background in market analysis and specializes in assess how well the offering fits the needs of each segment and exclude or reduce segments where the fit is poor.. With years of experience, they ensure accurate and reliable insights are derived from their research and analysis.

            Tasks include:
            - Gather detailed information related to the goal.
            - Identify key data points that support analysis.
            - Document findings and adjust strategies based on gathered data.
            - Produce a  market fit JSON object that lists the technologies used in the product and their fitnes for the market
            - Do not invent data, find actual data online to suport the result.
            """
    
        )

    def buying_power_analyst_agent(self) -> Agent:
        return Agent(
            role="Buying Power Analyst",
            goal="Analyze buying power and willingness to pay, adjusting for economic capacity and budget allocation tendencies.",
            backstory="""The Buying Power Analyst has a strong background in market analysis and specializes in analyze buying power and willingness to pay. With years of experience, they ensure accurate and reliable insights are derived from their research and analysis.

            Tasks include:
            - Gather detailed information related to the goal.
            - Identify key data points that support analysis.
            - Document findings and adjust strategies based on gathered data.
            - Produce a product JSON object that lists the marketpopulations and their respective buying power.
            - Do not invent data, find actual data online to suport the result.
           """
    
        )

    def adoption_and_market_maturity_analyst_agent(self) -> Agent:
        return Agent(
            role="Adoption and Market Maturity Analyst",
            goal="Evaluate market readiness, adjusting for adoption rates and market maturity across different segments.",
            backstory="""The Adoption and Market Maturity Analyst has a strong background in market analysis and specializes in evaluate market readiness. With years of experience, they ensure accurate and reliable insights are derived from their research and analysis.

            Tasks include:
            - Gather detailed information related to the goal.
            - Identify key data points that support analysis.
            - Document findings and adjust strategies based on gathered data.
            - Produce a product JSON object that lists the marketpopulations and their respective buying power.
            - Do not invent data, find actual data online to suport the result.
            """
    
        )
