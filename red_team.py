from deepteam import red_team
from deepteam.vulnerabilities import Bias
from deepteam.attacks.single_turn import PromptInjection

async def model_callback(input: str) -> str:
    # Replace this with your LLM application
    return f"I'm sorry but I can't answer this: {input}"

bias = Bias(types=["race"])
prompt_injection = PromptInjection()

red_team(model_callback=model_callback, vulnerabilities=[bias], attacks=[prompt_injection])