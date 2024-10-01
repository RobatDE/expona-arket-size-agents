from datetime import datetime
from typing import Callable
from langchain_openai import ChatOpenAI
from agents import TAMAgents
from job_manager import append_event
from crewai import Crew
import os

class TAMResearchCrew:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.crew = None
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "dolphin-2.9.2-qwen2-72b") 
        self.apiKey = os.getenv("OPENAI_API_KEY", "<empty key>") 
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
        self.goals = {}

    def setup_crew(self, companies: list[str], regions: list[str], products: list[str]):
        agents = TAMAgents()

        # Define agents
        research_manager = agents.research_manager(self.goals)
        product_definition_agent = agents.product_definition_analyst_agent(products)
        market_segmentation_agent = agents.market_segmentation_analyst_agent()
        geographic_constraints_agent = agents.geographic_constraints_analyst_agent()
        technological_limitations_agent = agents.technological_limitations_analyst_agent()
        economic_considerations_agent = agents.economic_considerations_analyst_agent()
        product_market_fit_agent = agents.product_market_fit_analyst_agent()
        buying_power_agent = agents.buying_power_analyst_agent()
        adoption_and_market_maturity_agent = agents.adoption_and_market_maturity_analyst_agent()

        # Define tasks for each agent with input data
        tasks = [
                    {
                        "agent": product_definition_agent,
                        "data": "product_details",
                        "description": "Analyze and define the product based on provided details.",
                        "expected_output": "A detailed product definition report."
                    },
                    {
                        "agent": market_segmentation_agent,
                        "data": "market_segments",
                        "description": "Identify relevant market segments.",
                        "expected_output": "A list of segmented markets based on demographics and psychographics."
                    },
                    {
                        "agent": geographic_constraints_agent,
                        "data": "geography",
                        "description": "Assess the geographic constraints of the market.",
                        "expected_output": "A detailed analysis of the geographic regions that can be served."
                    },
                    {
                        "agent": technological_limitations_agent,
                        "data": "technology",
                        "description": "Evaluate the technological limitations relevant to the product.",
                        "expected_output": "A report detailing the required technological adoption for the product."
                    },
                    {
                        "agent": economic_considerations_agent,
                        "data": "economic_factors",
                        "description": "Assess the economic factors related to the market.",
                        "expected_output": "An analysis of economic viability and price ranges."
                    },
                    {
                        "agent": product_market_fit_agent,
                        "data": "market_fit",
                        "description": "Evaluate the product-market fit for different segments.",
                        "expected_output": "A report on the product-market fit for each segment."
                    },
                    {
                        "agent": buying_power_agent,
                        "data": "buying_power",
                        "description": "Analyze the buying power of potential customers.",
                        "expected_output": "A report on the buying power and willingness to pay."
                    },
                    {
                        "agent": adoption_and_market_maturity_agent,
                        "data": "adoption_rates",
                        "description": "Assess the adoption rates and market maturity.",
                        "expected_output": "A detailed analysis of adoption rates and market maturity."
                    }
            ]


        # Extract goals from agents and create self.goals
        self.goals = [task['agent'].goal for task in tasks]

        # Setup crew with all agents and their tasks
        self.crew = Crew(
            agents=[
                research_manager,
                product_definition_agent,
                market_segmentation_agent,
                geographic_constraints_agent,
                technological_limitations_agent,
                economic_considerations_agent,
                product_market_fit_agent,
                buying_power_agent,
                adoption_and_market_maturity_agent,
            ],
            tasks=tasks,
            verbose=2
        )

    def kickoff(self):
            if not self.crew:
                append_event(self.job_id, "Crew not set up")
                return "Crew not set up"

                append_event(self.job_id, "Task Started")
            try:
                results = self.crew.kickoff()
                append_event(self.job_id, "Task Complete")
                return results
            except Exception as e:
                append_event(self.job_id, f"An error occurred: {e}")
                return str(e)
