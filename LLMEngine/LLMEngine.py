import time
import asyncio
from typing import List
from .llm_compiler import LLMCompiler
from .tools.tool_generator import ToolGenerator


class LLMEngine:
    def __init__(  # Fixed: was _init_ (missing underscores)
        self,
        llm,
        message_manager=None,
        memory=[],
        max_replan=3,
        verbose=False,
    ):
        self.message_manager = message_manager
        self.llm = llm
        self.memory = memory
        self.max_replan = max_replan
        self.verbose = verbose
        self.is_tables = False
        self.agent = LLMCompiler(
            name="LLMCompiler",  # Required by Chain base class
            llm=self.llm,
            max_replans=self.max_replan,
            benchmark=False,
            message_manager=self.message_manager
        )

    async def arun_and_time(self, func, *args, **kwargs):
        """Helper function to run and time a function.
        Raises exceptions to caller for proper error handling.
        """
        start = time.time()
        try:
            result = await func(*args, **kwargs)
        except Exception as e:
            print(f"Error: {e}")
            # Re-raise the exception instead of returning "ERROR"
            # This allows proper error handling in the calling code
            raise
        end = time.time()
        return result, end - start

    async def run(self, question , purpose ,tools ,instructions ,query_understanding="" ,temporal_context="" ,research_approach="" ,dos="" ,donts="" ,meta_example="" ,planner_example_prompt = "" , joinner_prompt ="",tool_path=None ):
        """
        Runs the compiler asynchronously
        
        Args:
        planner_example_prompt
        planner_example_prompt_replan
        joinner_prompt
        joinner_prompt_final
        input
        purpose
        instructions
        tools = [{class:search_tool ,name:fb_search, domains :["website1",],extra_info:"searches over facebook"}]]
        """
        # Create the input dictionary

        input_dict = {"input": question}
        input_dict["is_table_format"] = self.is_tables
        input_dict["planner_example_prompt"]=planner_example_prompt
        input_dict["planner_example_prompt_replan"]=None
        input_dict["joinner_prompt"]=joinner_prompt
        input_dict["joinner_prompt_final"]=None
        input_dict["purpose"]=purpose 
        input_dict["instructions"]=instructions
        input_dict["query_understanding"]=query_understanding  
        input_dict["temporal_context"]=temporal_context 
        input_dict["research_approach"]=research_approach 
        input_dict["dos"]=dos 
        input_dict["donts"]=donts 
        input_dict["meta_example"]=meta_example
        
        
        #tools_list= [{class:search_tool ,name:fb_search, domains :["website1",],extra_info:"searches over facebook"}]]
        input_dict["tools"]=ToolGenerator(tools,tool_path)
        print("using tools",input_dict["tools"])

        if self.is_tables:
            table_context = f"Answer this question: {question}"
            input_dict["input"] = f"{table_context}"
        
        # Call the agent with our modified input
        result, _ = await self.arun_and_time(
            self.agent.acall,
            input_dict,
            callbacks=None,
        )
        print("initial : ",result)
        if isinstance(result, dict):
            raw_answer = result.get(self.agent.output_key, "")
        else:
            raw_answer = str(result)
        if isinstance(result, str):
            thinking_process = result
        else:
            thinking_process = result.get("thinking_process", "")
                
        return raw_answer, [], thinking_process  # Fixed: removed self.data_loader.macro_sources since data_loader is not defined
    
