from typing import List
import os
from crewai import Agent
from langchain_openai import ChatOpenAI
from crewai_tools import SerperDevTool
from tools.youtube_search_tools import YoutubeVideoSearchTool

# market_size_agent, company_size_agent, market_penetration_agent, tam_final_agent

class MarketSizeResearchAgents():

    def __init__(self):
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4") 
        self.llm = ChatOpenAI(model=self.model_name)

    def research_manager(self, companies: List[str], regions: List[str]) -> Agent:
        return Agent(
            role="Company Market Size Research Manager",
            goal=f"""Generate a list of JSON objects containing the data to accuratelyt dscibe the company named in companies,
                     The regions data is used to limit the scpoe of the research
             
                Companies: {companies}
                Regions: {regions}

                The company json object should include all demographic data for the company and its products and employee count 

                Important:
                - The final list of JSON objects must include all companies and regions. Do not leave any out.
                - If you can't find information for a specific position, fill in the information with the word "MISSING".
                - Do not generate fake information. Only return the information you find. Nothing else!
                - Do not stop researching until you find the requested information for each position in each company.
                - All the companies and positions exist so keep researching until you find the information for each one.
                - Make sure you each researched position for each company contains 3 blog articles and 3 YouTube interviews.
                """,
            backstory="""As a Company Research Manager, you are responsible for aggregating all the researched information
                into a list.""",
            llm=self.llm,
            tools=[self.searchInternetTool],
            verbose=True,
            allow_delegation=True
        )

    def company_research_agent(self) -> Agent:
        return Agent(
            role="Company Research Agent",
            goal="""Look up the specific regions for a given company and find urls for 3 recent marketing sizing reports. It is your job to return this collected 
                information in a JSON object""",
            backstory="""As a Company Research Agent, you are responsible for looking up specific details  
                about a company and gathering relevant information.
                
                Important:
                - Once you've found the information, immediately stop searching for additional information.
                - Only return the requested information. NOTHING ELSE!
                - The informatoin should include democgraphics, products, company size and regions served
                - Do not generate fake information. Only return the information you find. Nothing else!
                """,
            tools=[self.searchInternetTool],
            llm=self.llm,
            verbose=True
        )
    
    def company_tam_industry_agent(self) -> Agent:
        return Agent(
            role="Total Addressable Market industry analyst",
            goal="""Define the target market for the company and region""",
            backstory="""Important:
                 - Identify the industry, product/service, and customer segments you want to serve.
                 - Determine the geographic scope (e.g., global, regional, national).
                 - Clearly define the target customer profile (e.g., enterprise, SMB, consumer).
                 - Do not generate fake information. Only return the information you find. Nothing else!
                 """,
            tools=[self.searchInternetTool],
            llm=self.llm,
            verbose=True
        )                  
      
    def company_tam_market_reports_agent(self) -> Agent:
        return Agent(
            role="Total Addressable Market research agent",
            goal="""Gather market sizing data about the TAM :""",
            backstory="""Important:
                    Identify reputable industry reports, analyst estimates, and credible secondary sources that provide market size data.
                    Look for information on the total number of potential customers, average customer spending, growth rates, etc.
                    Ensure the data is as recent and relevant to your target market as possible.
                    """,
            tools=[self.searchInternetTool],
            llm=self.llm,
            verbose=True
        )

    def company_tam_total_customers_agent(self) -> Agent:
        return Agent(
            role="Total Addressable Market research agent",
            goal="""Gather data about the total number of potential customer for the companies products in the specified regoin:""",
            backstory="""Important:
                    Determine the total number of potential customers:
                    Estimate the total number of companies, individuals, or households that could potentially use the companies product/service.
                    This may involve researching industry associations, trade publications, government statistics, and other authoritative sources.
                    To understand the composition, break down the total market into relevant segments (e.g., by company size, industry, and demographics).
                 - Clearly define the target customer profile (e.g., enterprise, SMB, consumer).
                 - Do not generate fake information. Only return the information you find. Nothing else!""",
            tools=[self.searchInternetTool],
            llm=self.llm,
            verbose=True
        ) 
    
    def company_tam_customer_spend_agent(self) -> Agent:
        return Agent(
            role="Total Addressable Market research agent",
            goal="""Calculate the average customer spending for similar products and services""",

            backstory="""Important:
                    Gather data on the typical annual or lifetime value of a customer in your target market.
                    This may involve analyzing industry benchmarks, competitor pricing, or conducting customer surveys.
                    Consider factors like average transaction size, frequency of purchases, and ancillary revenue streams.
                 - Clearly define the target customer profile (e.g., enterprise, SMB, consumer).
                 - Do not generate fake information. Only return the information you find. Nothing else!""",
            tools=[self.searchInternetTool],
            llm=self.llm,
            verbose=True
        )
    
    def company_tam_penetration_agent(self) -> Agent:
        return Agent(
            role="Total Addressable Market research agent",
            goal="""Estimate the market penetration rate:""",

            backstory="""Important:
                    Determine the realistic adoption rate or market penetration you can achieve with your product/service.
                    This will depend on factors like market maturity, competition, pricing, and your company's go-to-market strategy.
                    Use industry benchmarks, historical data, and expert insights to inform your penetration rate assumptions.
                 - Clearly define the target customer profile (e.g., enterprise, SMB, consumer).
                 - Do not generate fake information. Only return the information you find. Nothing else!""",
            tools=[self.searchInternetTool],
            llm=self.llm,
            verbose=True
        )
   
    def company_tam_final_agent(self) -> Agent:
        return Agent(
            role="Total Addressable Market research agent",
            goal="""Calculate the Total Addressable Market:""",
            backstory="""Important:
                    The formula for TAM is: TAM = Total number of potential customers × Average customer spending × Market penetration rate
                    For example:
                        Total potential customers: 1 million
                        Average customer spending: $500 per year
                        Market penetration rate: 20%
                        TAM = 1,000,000 × $500 × 0.20 = $100,000,000
                        Validate and refine the TAM estimate:
                        Cross-check your TAM calculation with multiple data sources and market research.
                        Adjust your assumptions based on new information or changes in the market.
                        Conduct sensitivity analysis to understand the impact of varying key input parameters.
                   Present the TAM estimate:
                    Document your assumptions, data sources, and the step-by-step calculation.
                    Present the TAM as a range rather than a single-point estimate to account for uncertainties.
                    Provide context and analysis around the TAM figure, such as market trends, growth projections, and competitive landscape.
                 - Use the included example of TAM formula as the basis of your calculatoin but use real data from research.
                 - Do not generate fake information. Only return the information you find. Nothing else!""",
            tools=[self.searchInternetTool],
            llm=self.llm,
            verbose=True
        )               
