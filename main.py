import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from dataclasses import dataclass
from typing import Optional, List
import asyncio
from browser_use import Agent
import os

load_dotenv()


@dataclass
class ActionResult:
	is_done: bool
	extracted_content: Optional[str]
	error: Optional[str]
	include_in_memory: bool


@dataclass
class AgentHistoryList:
	all_results: List[ActionResult]
	all_model_outputs: List[dict]


def parse_agent_history(history_str: str) -> None:
	console = Console()

	# Split the content into sections based on ActionResult entries
	sections = history_str.split('ActionResult(')

	for i, section in enumerate(sections[1:], 1):  # Skip first empty section
		# Extract relevant information
		content = ''
		if 'extracted_content=' in section:
			content = section.split('extracted_content=')[1].split(',')[0].strip("'")

		if content:
			header = Text(f'Step {i}', style='bold blue')
			panel = Panel(content, title=header, border_style='blue')
			console.print(panel)
			console.print()


async def run_browser_task(
	task: str,
	api_key: str,
	model: str = 'gpt-4o',
	headless: bool = True,
) -> AgentHistoryList:
	env_api_key = os.getenv('OPENAI_API_KEY')
	if env_api_key:
		api_key = env_api_key
	elif not api_key.strip():
		return AgentHistoryList(all_results=[], all_model_outputs=[])

	try:
		agent = Agent(
			task=task,
			llm=ChatOpenAI(model='gpt-4o'),
		)
		result = await agent.run()
		
		# Assuming result is a list of dictionaries
		all_results = []
		all_model_outputs = []

		# Example parsing logic (adjust based on actual result structure)
		for res in result:
			action_result = ActionResult(
				is_done=res.get('is_done', False),
				extracted_content=res.get('extracted_content', None),
				error=res.get('error', None),
				include_in_memory=res.get('include_in_memory', False)
			)
			all_results.append(action_result)
			all_model_outputs.append(res)  # or some specific part of res

		return AgentHistoryList(all_results=all_results, all_model_outputs=all_model_outputs)
	except Exception as e:
		return AgentHistoryList(
			all_results=[ActionResult(is_done=False, extracted_content=None, error=str(e), include_in_memory=False)],
			all_model_outputs=[]
		)


def format_agent_output(agent_history: AgentHistoryList) -> str:
	formatted_output = "# Task Results\n\n"
	
	for i, result in enumerate(agent_history.all_results, start=1):
		if result.extracted_content:
			formatted_output += f"## Result {i}\n{result.extracted_content}\n\n"
		if result.error:
			formatted_output += f"**Error**: {result.error}\n\n"
	
	return formatted_output


def create_ui():
	custom_css = """
	body {
		background-color: #0d0d0d;
		color: #e0e0e0;
		font-family: 'Courier New', monospace;
	}
	.gr-button {
		background-color: #ff00ff;
		color: #0d0d0d;
		border: none;
		cursor: pointer;
	}
	.gr-textbox, .gr-dropdown, .gr-checkbox {
		background-color: #0d0d0d;
		color: #e0e0e0;
		border: 1px solid #00ffff;
	}
	"""

	with gr.Blocks(title='Browser Use GUI', css=custom_css) as interface:
		gr.Markdown(
			"""
			<div style="text-align: center; font-size: 24px; color: #e0e0e0;">
				<h1>Gopher It!</h1>
			</div>
			"""
		)

		with gr.Row():
			with gr.Column(scale=1, min_width=300):
				api_key = gr.Textbox(
					label='OpenAI API Key', 
					placeholder='sk-...', 
					type='password'
				)
				task = gr.Textbox(
					label='Task Description',
					placeholder='E.g., Find flights from New York to London for next week',
					lines=3
				)
				model = gr.Dropdown(
					choices=['gpt-4', 'gpt-3.5-turbo'], 
					label='Model', 
					value='gpt-4'
				)
				headless = gr.Checkbox(
					label='Run Headless', 
					value=True
				)
				submit_btn = gr.Button('Run Task')

			with gr.Column(scale=2, min_width=500):
				output = gr.Markdown(
					label='Output'
				)
				image_output = gr.Image(
					label='Agent History GIF'
				)

		def on_submit(task, api_key, model, headless):
			agent_history = asyncio.run(run_browser_task(task, api_key, model, headless))
			gif_path = "agent_history.gif"  # Path to the pre-saved GIF
			return format_agent_output(agent_history), gif_path

		submit_btn.click(
			fn=on_submit,
			inputs=[task, api_key, model, headless],
			outputs=[output, image_output],
		)

	return interface


if __name__ == '__main__':
	demo = create_ui()
	demo.launch(pwa=True)