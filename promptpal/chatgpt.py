
from propmptpal.core import CreateAgent
from openai import OpenAI

# Confirm environment API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise EnvironmentError("OPENAI_API_KEY environment variable not found!")

# Initialize OpenAI client and conversation thread
client = OpenAI(api_key=api_key)
thread = client.beta.threads.create()
thread.current_thread_calls = 0
client.thread_ids = set([thread.id])
total_cost = 0.0
total_tokens = {}


class ChatGPTAgent(CreateAgent):
	    """
    A handler for managing queries to the OpenAI API, including prompt preparation,
    API request submission, response processing, and logging.

    This class provides a flexible interface to interact with OpenAIs models, including
    text-based models (e.g., GPT-4) and image generation models (e.g., DALL-E). It supports
    features such as associative prompt refinement, chain-of-thought reasoning, code extraction,
    logging, unit testing, and much more.

    Attributes:
        model (str): The model to use for the query (e.g., 'gpt-4o-mini', 'dall-e-3').
        verbose (bool): If True, prints detailed logs and status messages.
        silent (bool): If True, silences all StdOut messages.
        refine (bool): If True, refines the prompt before submission.
        chain_of_thought (bool): If True, enables chain-of-thought reasoning.
        save_code (bool): If True, extracts and saves code snippets from the response.
        scan_dirs (bool): If True, recursively scans directories found in prompt for existing files, extracts contents, and adds to prompt.
        logging (bool): If True, logs the session to a file.
        seed (int or str): Seed for reproducibility. Can be an integer or a string converted to binary.
        iterations (int): Number of response iterations for refining or condensing outputs.
        dimensions (str): Dimensions for image generation (e.g., '1024x1024').
        quality (str): Quality setting for image generation (e.g., 'hd').
        role (str): The role or persona for the query (e.g., 'assistant', 'artist').
        tokens (dict): Tracks token usage for prompt and completion.
        prefix (str): A unique prefix for log files and outputs.
        client (OpenAI): The OpenAI client instance for API requests.
        glyph (bool): If True, restructures queries with representative/associative glyphs and logic flow
        temperature (float): Range from 0.0 to 2.0, lower values increase randomness, and higher values increase randomness.
        top_p (float): Range from 0.0 to 2.0, lower values increase determinism, and higher values increase determinism.
        message_limit (int): Maximum number of messages to a single thread before summarizing content and passing to new instance
        last_message (str): Last returned system message
        thread_id (str): Current conversation thread

    Current role shortcuts:
        assistant: Standard personal assistant with improved ability to help with tasks
        developer: Generates complete, functional application code based on user requirements, ensuring clarity and structure.
        prompt: Specializes in analyzing and refining AI prompts to enhance clarity, specificity, and effectiveness without executing tasks.
        refactor: Senior full stack developer with emphases in correct syntax and documentation
        tester: Quality assurance tester with experience in software testing and debugging, generates high-quality unit tests
        analyst: For structured data analysis tasks, adhering to strict validation rules, a detailed workflow, and professional reporting
        visualize: Create clear, insightful data visualizations and provide analysis using structured formats, focusing solely on visualization requests and recommendations.
        writer: Writing assistant to help with generating science & technology related content
        editor: Text editing assistant to help with clarity and brevity
        artist: Creates an images described by the prompt, default style leans toward illustrations
        photographer: Generates more photo-realistic images

    Methods:
        __init__: Initializes the handler with default or provided values.
        request: Submits a query to the OpenAI API and processes the response.
        status: Reports current attributes and status of agent and session information 
        cost_report: Reports spending information
        token_report: Reports token generation information
        thread_report: Report active threads from current session
        start_new_thread: Start a new thread with only the current agent.
        summarize_current_thread: Summarize current conversation history for future context parsing.
        _extract_and_save_code: Extracts code snippets from the response and saves them to files.
        _setup_logging: Prepares logging setup.
        _prepare_query_text: Prepares the query, including prompt modifications and image handling.
        _validate_model_selection: Validates and selects the model based on user input or defaults.
        _prepare_system_role: Selects the role based on user input or defaults.
        _append_file_scanner: Scans files in the message and appends their contents.
        _validate_image_params: Validates image dimensions and quality for the model.
        _handle_text_request: Processes text-based responses from OpenAIs chat models.
        _handle_image_request: Processes image generation requests using OpenAIs image models.
        _condense_iterations: Condenses multiple API responses into a single coherent response.
        _refine_user_prompt: Refines an LLM prompt using specified rewrite actions.
        _update_token_count: Updates token count for prompt and completion.
        _log_and_print: Logs and prints the provided message if verbose.
        _calculate_cost: Calculates the approximate cost (USD) of LLM tokens generated.
        _string_to_binary: Converts a string to a binary-like variable for use as a random seed.
    """

    def __init__(self, model = "gpt-4o-mini"):

    	self._validate_model_selection(model)

    	# Agent-specific thread params
        global thread
        self.thread_id = thread.id
        thread.message_limit = message_limit
        if self.new_thread == True:
            self.start_new_thread()

        # Update token counters
        global total_tokens
        if self.model not in total_tokens.keys():
            total_tokens[self.model] = {"prompt": 0, "completion": 0}

    def _validate_model_selection(self, input_model):
        """Validates and selects the model based on user input or defaults."""
        openai_models = ["gpt-4o","o1","o1-mini","o1-preview","dall-e-3","dall-e-2"]
        self.model = input_model.lower() if input_model.lower() in openai_models else "gpt-4o-mini"

    def start_new_thread(self, context=None):
        """Start a new thread with only the current agent and adds previous context if needed."""
        global thread
        thread = client.beta.threads.create()
        thread.current_thread_calls = 0
        thread.message_limit = self.message_limit

        # Add previous context
        if context:
            previous_context = client.beta.threads.messages.create(
                thread_id=thread.id, role="user", content=context)

        global client
        client.thread_ids |= set([thread.id])
        self.thread_id = thread.id

        # Report
        self._log_and_print(f"New thread created and added to current agent: {self.thread_id}\n", 
            self.verbose, self.logging)
        
    def _init_chat_completion(self, prompt, model='gpt-4o-mini', role='user', iters=1, seed=42, temp=0.7, top_p=1.0):
        """Initialize and submit a single chat completion request"""
        message = [{"role": "user", "content": prompt}, {"role": "system", "content": role}]

        completion = client.chat.completions.create(
            model=model, messages=message, n=iters,
            seed=seed, temperature=temp, top_p=top_p)
        self._update_token_count(completion)
        self._calculate_cost()

        return completion

    def _update_token_count(self, response_obj):
        """Updates token count for prompt and completion."""
        global total_tokens
        total_tokens[self.model]["prompt"] += response_obj.usage.prompt_tokens
        total_tokens[self.model]["completion"] += response_obj.usage.completion_tokens
        # Agent-specific counts
        self.tokens["prompt"] += response_obj.usage.prompt_tokens
        self.tokens["completion"] += response_obj.usage.completion_tokens

    def _calculate_cost(self, dec=5):
        """Calculates approximate cost (USD) of LLM tokens generated to a given decimal place"""
        global total_cost

        # As of January 24, 2025
        rates = {
            "gpt-4o": (2.5, 10),
            "gpt-4o-mini": (0.150, 0.600),
            "o1-mini": (3, 12),
            "o1-preview": (15, 60),
            "dall-e-3": (2.5, 0.040),
            "dall-e-2": (2.5, 0.040),
        }
        if self.model in rates:
            prompt_rate, completion_rate = rates.get(self.model)
            prompt_cost = round((self.tokens["prompt"] * prompt_rate) / 1e6, dec)
            completion_cost = round((self.tokens["completion"] * completion_rate) / 1e6, dec)
        else:
            prompt_cost = completion_cost = 0.0

        total_cost += round(prompt_cost + completion_cost, dec)
        self.cost["prompt"] += prompt_cost
        self.cost["completion"] += completion_cost

    def _refine_custom_role(self, init_role):
        """Reformat input custom user roles for improved outcomes."""

        self._log_and_print(f"Refining custom role text...\n", self.verbose, self.logging)

        # Reformat role text
        refine_prompt = "Format and improve the following system role propmt to maximize clarity and potential output quality:\n\n" + init_role
        response = self._init_chat_completion(refine_prompt)
        custom_role = response.choices[0].message.content.strip()
        
        # Name custom role
        refine_prompt = "Generate a short and accurate name for the following system role prompt:\n\n" + custom_role
        response = self._init_chat_completion(refine_prompt)
        role_name = response.choices[0].message.content.strip()

        reportStr = f"""Role name:
{role_name}

Description:
{custom_role}

"""
        self._log_and_print(reportStr, self.verbose, self.logging)

        return role_name, custom_role

    def thread_report(self):
        """Report active threads from current session"""
        threadStr = f"""Current session threads:
    {'\n\t'.join(client.thread_ids)}
"""
        self._log_and_print(threadStr, True, self.logging)

    def _condense_iterations(self, api_response):
        """Condenses multiple API responses into a single coherent response."""
        api_responses = [r.message.content.strip() for r in api_response.choices]
        api_responses =  "\n\n".join(
            ["\n".join([f"Iteration: {i + 1}", api_responses[i]])
            for i in range(len(api_responses))])

        self._log_and_print(
            f"\nAgent using gpt-4o-mini to condense system responses...", self.verbose, self.logging
        )
        condensed = self._init_chat_completion( 
            prompt= modifierDict['condense'] + "\n\n" + api_responses, 
            iters=self.iterations, seed=self.seed)

        message = condensed.choices[0].message.content.strip()
        self._log_and_print(
            f"\nCondensed text:\n{message}", self.verbose, self.logging
        )

        return message

    def _refine_user_prompt(self, old_prompt):
        """Refines an LLM prompt using specified rewrite actions."""
        updated_prompt = old_prompt
        if self.refine_prompt == True:
            actions = set(["expand", "amplify"])
            actions |= set(
                re.sub(r"[^\w\s]", "", word).lower()
                for word in old_prompt.split()
                if word.lower() in refineDict
            )
            action_str = "\n".join(refineDict[a] for a in actions) + "\n\n"
            updated_prompt = modifierDict["refine"] + action_str + old_prompt

        if self.glyph_prompt == True:
            updated_prompt += modifierDict["glyph"]

        refined = self._init_chat_completion(
            prompt=updated_prompt, 
            role=self.role,
            seed=self.seed, 
            iters=self.iterations,
            temp=self.temperature, 
            top_p=self.top_p)

        if self.iterations > 1:
            new_prompt = self._condense_iterations(refined)
        else:
            new_prompt = refined.choices[0].message.content.strip()

        self._log_and_print(
            f"Refined query prompt:\n{new_prompt}", self.verbose, self.logging)

        return new_prompt

    def _create_new_agent(self, interpreter=False):
        """
        Creates a new assistant based on user-defined parameters

        Args:
            interpreter (bool): Whether to enable the code interpreter tool.

        Returns:
            New assistant assistant class instance
        """
        try:
            agent = client.beta.assistants.create(
                name=self.role_name,
                instructions=self.role,
                model=self.model,
                tools=[{"type": "code_interpreter"}] if interpreter == True else [])
            self.agent = agent.id
        except Exception as e:
            raise RuntimeError(f"Failed to create assistant: {e}")

    def _run_thread_request(self) -> str:
        """
        Sends a user prompt to an existing thread, runs the assistant, 
        and retrieves the response if successful.
        
        Returns:
            str: The text response from the assistant.
        
        Raises:
            ValueError: If the assistant fails to generate a response.
        """
        # Adds user prompt to existing thread.
        try:
            new_message = client.beta.threads.messages.create(
                thread_id=self.thread_id, role="user", content=self.prompt)
        except Exception as e:
            raise RuntimeError(f"Failed to create message: {e}")

        # Run the assistant on the thread
        current_run = client.beta.threads.runs.create(
            thread_id=self.thread_id,
            assistant_id=self.agent)

        # Wait for completion and retrieve responses
        while True:
            self.run_status = client.beta.threads.runs.retrieve(thread_id=self.thread_id, run_id=current_run.id)
            if self.run_status.status in ["completed", "failed"]:
                break
            else:
                time.sleep(1)  # Wait before polling again

        if self.run_status.status == "completed":
            messages = client.beta.threads.messages.list(thread_id=self.thread_id)
            if messages.data:  # Check if messages list is not empty
                return messages.data[0].content[0].text.value
            else:
                raise ValueError("No messages found in the thread.")
        else:
            raise ValueError("Assistant failed to generate a response.")
