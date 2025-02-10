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
from datetime import datetime

# Load environment variables from .env file if present
load_dotenv()

# Retrieve the API key from the environment
api_key = os.getenv('OPENAI_API_KEY')

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
	api_key: str = api_key,  # Default to the environment variable
	model: str = 'gpt-4o',
	headless: bool = True,
) -> AgentHistoryList:
	if not api_key:
		return AgentHistoryList(all_results=[], all_model_outputs=[])

	try:
		agent = Agent(
			task=task,
			llm=ChatOpenAI(model='gpt-4o', api_key=api_key),
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
				<img src="gopherit.png" alt="Gopher It Logo" style="width:100px;height:auto;">
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
			# Print the API key
			print(f"API Key: {api_key}")

			# Record the time when the task is submitted
			submit_time = datetime.now()
			print(f"Task submitted at: {submit_time}")

			# Run the task
			try:
				agent_history = asyncio.run(run_browser_task(task, api_key, model, headless))
				print("Task completed successfully.")
			except Exception as e:
				print(f"Error running task: {e}")
				return f"Error: {e}", None

			# Save the GIF with a timestamp
			timestamp = submit_time.strftime("%Y%m%d%H%M%S")
			gif_path = f"agent_history_{timestamp}.gif"
			print(f"GIF saved as: {gif_path}")

			# Check if the current time is after the submit time
			if datetime.now() > submit_time:
				return format_agent_output(agent_history), gif_path
			else:
				# If no GIF is available, return the active output
				return format_agent_output(agent_history), None

		submit_btn.click(
			fn=on_submit,
			inputs=[task, api_key, model, headless],
			outputs=[output, image_output],
		)

	return interface


if __name__ == '__main__':
	demo = create_ui()
	demo.launch(server_name="0.0.0.0", server_port=7860, pwa=True)