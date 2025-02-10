import os
import re
import sys
import time
import string
import requests
from datetime import datetime
from collections import defaultdict

from promptpal import utils
from promptpal.lib import text_library

total_cost = 0.0
total_tokens = {}


# Base class
class CreateAgent:
    """
    A handler for managing queries to the OpenAI API, including prompt preparation,
    API request submission, response processing, and logging.

    This class provides a flexible interface to interact with OpenAIs models, including
    text-based models (e.g., GPT-4) and image generation models (e.g., DALL-E). It supports
    features such as associative prompt refinement, chain-of-thought reasoning, code extraction,
    logging, and unit testing.

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
        message_limit (int): Maximum number of messages to a single chat before summarizing content and passing to new instance
        last_message (str): Last returned system message

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
        chat: Submits a query to the API and processes the response.
        status: Reports current attributes and status of agent and session information
        cost_report: Reports spending information
        token_report: Reports token generation information
        chat_report: Report active chats from current session
        start_new_chat: Start a new chat with only the current agent.
        summarize_current_chat: Summarize current conversation history for future context parsing.
        _prepare_query_text: Prepares the query, including prompt modifications and image handling.
        _validate_model_selection: Validates and selects the model based on user input or defaults.
        _prepare_system_role: Selects the role based on user input or defaults.
        _validate_image_params: Validates image dimensions and quality for the model.
        _handle_text_request: Processes text-based responses from chat models.
        _handle_image_request: Processes image generation requests using image models.
        _condense_iterations: Condenses multiple API responses into a single coherent response.
        _refine_user_prompt: Refines an LLM prompt using specified rewrite actions.
        _update_token_count: Updates token count for prompt and completion.
        _calculate_cost: Calculates the approximate cost (USD) of LLM tokens generated.
    """

    def __init__(
        self,
        logging=True,
        verbose=True,
        silent=False,
        refine=False,
        glyph=False,
        chain_of_thought=False,
        save_code=False,
        scan_dirs=False,
        new_chat=False,
        role="assistant",
        seed="t634e``R75T86979UYIUHGVCXZ",
        iterations=1,
        temperature=0.7,
        top_p=1.0,
        dimensions="NA",
        quality="NA",
        stage="normal",
        message_limit=20,
        **kwargs,
    ):
        """
        Initialize the handler with default or provided values.
        """
        self.logging = logging
        self.verbose = verbose
        self.silent = silent
        self.refine_prompt = refine
        self.glyph_prompt = glyph
        self.chain_of_thought = chain_of_thought
        self.save_code = save_code
        self.scan_dirs = scan_dirs
        self.new_chat = new_chat
        self.role = role
        self.seed = seed
        self.iterations = iterations
        self.temperature = temperature
        self.top_p = top_p
        self.dimensions = dimensions
        self.quality = quality
        self.stage = stage
        self.message_limit = message_limit

        # Check user input types
        self._validate_types()

        # Agent-specific chat params
        global chat
        self.chat_id = chat.id
        chat.message_limit = message_limit
        if self.new_chat == True:
            self.start_new_chat()

        # Update token counters
        global total_tokens
        self.cost = {"prompt": 0.0, "completion": 0.0}
        self.tokens = {"prompt": 0, "completion": 0}
        if self.model not in total_tokens.keys():
            total_tokens[self.model] = {"prompt": 0, "completion": 0}

        # Validate specific hyperparams
        self._validate_model_selection(self.model, kwargs["valid_models"])
        self.stage = self.stage if self.stage == "refine_only" else "normal"
        self.seed = (
            self.seed
            if isinstance(self.seed, int)
            else utils.string_to_binary(self.seed)
        )
        self.temperature, self.top_p = utils.validate_probability_params(
            self.temperature, self.top_p
        )

        # Validate user inputs
        self._prepare_system_role(role)
        if self.model in ["dall-e-2", "dall-e-3"]:
            self._validate_image_params(dimensions, quality)
        self._create_new_agent(interpreter=self.save_code)

        # Initialize reporting and related vars
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.prefix = f"{self.label}.{self.model.replace('-', '_')}.{self.timestamp}"
        if self.logging:
            self.log_file = utils.setup_logging(self.prefix)
        self._log_and_print(self.status(), False, self.logging)

    def _validate_types(self):
        """
        Validates the types of the instance attributes for CreateAgent.

        Raises:
            TypeError: If any attribute has an incorrect type.
            ValueError: If any integer attribute is not positive.
        """
        expected_types = {
            "model": str,
            "logging": bool,
            "verbose": bool,
            "silent": bool,
            "refine_prompt": bool,
            "glyph_prompt": bool,
            "chain_of_thought": bool,
            "save_code": bool,
            "scan_dirs": bool,
            "new_chat": bool,
            "model": str,
            "role": str,
            "seed": (int, str),  # seed can be either int or str
            "iterations": int,
            "temperature": float,
            "top_p": float,
            "dimensions": str,
            "quality": str,
            "stage": str,
            "message_limit": int,
        }

        for attr_name, expected_type in expected_types.items():
            value = getattr(self, attr_name, None)  # Get the attribute value from self
            if isinstance(expected_type, tuple):
                # Check if value matches any expected type in the tuple
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Expected type for {attr_name} is {expected_type}, got {type(value).__name__}"
                    )
            else:
                # Check if value matches the expected type
                if not isinstance(value, expected_type):
                    raise TypeError(
                        f"Expected type for {attr_name} is {expected_type}, got {type(value).__name__}"
                    )

            # Check if integer-type values are positive
            if expected_type == int and value <= 0:
                raise ValueError(f"{attr_name} must be a positive integer, got {value}")

    def _prepare_query_text(self, prompt_text):
        """
        Prepares the query, including prompt modifications and image handling.
        """
        self.prompt = prompt_text

        # Identifies files to be read in
        files = utils.find_existing_files(prompt_text)
        for f in files:
            self.prompt += "\n\n" + utils.read_file_contents(f)
        if self.scan_dirs == True:
            paths = utils.find_existing_paths(prompt_text)
            for d in paths:
                self.prompt += "\n\n" + utils.scan_directory(d)

        # Refine prompt if required
        if self.refine_prompt or self.glyph_prompt:
            self.prompt = self._refine_user_prompt(self.prompt)

    def _validate_model_selection(self, input_model, valid_models):
        """Validates and selects the model based on user input or defaults."""
        self.model = (
            input_model.lower()
            if input_model.lower() in valid_models
            else valid_models[0]
        )

    def _prepare_system_role(self, input_role):
        """Prepares system role text."""
        roleDict = text_library["roles"]
        # Selects the role based on user input or defaults.
        if input_role.lower() in roleDict:
            self.label = input_role.lower()
            builtin = roleDict[input_role.lower()]
            self.role = builtin["prompt"]
            self.role_name = builtin["name"]
        elif input_role.lower() in ["user", ""]:
            self.role = "user"
            self.label = "default"
            self.role_name = "Default User"
        else:
            self.role_name, self.role = self._refine_custom_role(input_role)
            self.label = "custom"

        # Add chain of thought reporting
        if self.chain_of_thought:
            self.role += text_library["modifiers"]["cot"]

    def _refine_custom_role(self, init_role):
        """Reformat input custom user roles for improved outcomes."""
        self._log_and_print(
            f"Refining custom role text...\n", self.verbose, self.logging
        )

        # Reformat role text
        refine_prompt = (
            "Format and improve the following system role propmt to maximize clarity and potential output quality:\n\n"
            + init_role
        )
        custom_role = self._init_chat_completion(refine_prompt)

        # Name custom role
        refine_prompt = "Generate a short and accurate name for the following system role prompt:\n\n"
        refine_prompt += custom_role
        refine_prompt += "\n\nReturn only the generated name."
        role_name = self._init_chat_completion(refine_prompt)

        reportStr = f"""Role name: {role_name}\n"""
        reportStr += f"""\nDescription:\n{custom_role}\n"""

        self._log_and_print(reportStr, self.verbose, self.logging)

        return role_name, custom_role

    def status(self):
        """Generate status message."""
        statusStr = f"""
Agent parameters:
    Model: {self.model}
    Role: {self.role_name}
    
    Chain-of-thought: {self.chain_of_thought}
    Prompt refinement: {self.refine_prompt}
    Associative glyphs: {self.glyph_prompt}
    Response iterations: {self.iterations}
    Subdirectory scanning: {self.scan_dirs}
    Text logging: {self.logging}
    Verbose StdOut: {self.verbose}
    Code snippet detection: {self.save_code}

    Image dimensions: {self.dimensions}
    Image quality: {self.quality}

    Time stamp: {self.timestamp}
    Seed: {self.seed}
    Assistant ID: {self.agent_id}
    Active chat ID: {self.chat_id}
    Requests in current chat: {chat.current_chat_messages}
"""
        self._log_and_print(statusStr, True, self.logging)

        # Token usage report
        self.token_report()
        # $$$ report
        self.cost_report()
        # Chat report
        self.chat_report()

    def chat(self, prompt=""):
        """Submits the query to OpenAIs API and processes the response."""
        # Checks for last system response is not prompt provided
        if prompt == "":
            try:
                prompt = self.last_message
            except Exception as e:
                raise ValueError(f"No existing messages found in chat: {e}")

        # Update user prompt
        self._prepare_query_text(prompt)
        self._log_and_print(
            f"\n{self.role_name} using {self.model} to process continued conversation...\n",
            True,
            self.logging,
        )

        if self.stage != "refine_only":
            if "dall-e" not in self.model:
                chat.current_chat_messages += 1
                self._handle_text_request()
            else:
                self._handle_image_request()

        # Check current scope chat
        if chat.current_chat_messages >= chat.message_limit:
            self._log_and_print(
                f"\nReached end of current chat limit.\n", self.verbose, False
            )
            summary = self.summarize_current_chat()
            self.start_new_chat(
                "The following is a summary of a ongoing conversation with a user and an AI assistant:\n"
                + summary
            )

    def _log_and_print(self, message, verbose=True, logging=True):
        """Logs and prints the provided message if verbose."""
        if message:
            if verbose == True and self.silent == False:
                print(message)
            if logging == True:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")

    def summarize_current_chat(self):
        """Summarize current conversation history for future context parsing."""
        self._log_and_print(
            f"Using {self.small_model} to summarize current chat...\n",
            self.verbose,
            False,
        )

        # Get all chat messages
        all_messages = self._get_current_messages()

        # Generate concise summary
        summary_prompt = text_library["modifiers"]["summarize"] + "\n\n" + all_messages
        summarized = self._init_chat_completion(
            model=self.small_model,
            prompt=summary_prompt,
            iters=self.iterations,
            seed=self.seed,
        )

        return summarized

    def _handle_text_request(self):
        """Processes text-based responses from OpenAIs chat models."""
        self.last_message = self._send_chat_message()
        self._update_token_count(self.run_status)
        self._calculate_cost()
        self._log_and_print(self.last_message, True, self.logging)

        # Extract code snippets
        code_snippets = utils.extract_code_snippets(self.last_message)
        if self.save_code and len(code_snippets) > 0:
            self.code_files = []
            reportStr = "\nExtracted code saved to:\n"
            for lang in code_snippets.keys():
                code = code_snippets[lang]
                objects = utils.extract_object_names(code, lang)
                file_name = f"{utils.find_max_lines(code, objects)}.{self.timestamp}{text_library["extensions"].get(lang, f'.{lang}')}".lstrip(
                    "_."
                )
                file_name = _check_unique_filename(file_name)
                reportStr += f"\t{file_name}\n"
                self._write_script(code, file_name)

            self._log_and_print(reportStr, True, self.logging)

    def _write_script(self, content, file_name, outDir="code", lang=None):
        """Writes code to a file."""
        os.makedirs(outDir, exist_ok=True)
        self.code_files.append(f"{os.getcwd()}/{outDir}/{file_name}")
        with open(f"{outDir}/{file_name}", "w", encoding="utf-8") as f:
            if lang:
                f.write(f"#!/usr/bin/env {lang}\n\n")
            f.write(f"# Code generated by {self.model}\n\n")
            f.write(content)

    def token_report(self):
        """Generates session token report."""
        allTokensStr = ""
        for x in total_tokens.keys():
            allTokensStr += f"{x}: Input = {total_tokens[x]['prompt']}; Completion = {total_tokens[x]['completion']}\n"

        tokenStr = "Overall session tokens:\n\t" + allTokensStr
        tokenStr += "\tCurrent agent tokens:\n"
        tokenStr += f"\tInput: {self.tokens['prompt']}\n"
        tokenStr += f"\tOutput: {self.tokens['completion']}\n"

        self._log_and_print(tokenStr, True, self.logging)

    def chat_report(self):
        """Report active chats from current session"""
        chatStr = "Current session chat IDs:\n" + "\n\t".join(client.chat_ids)
        self._log_and_print(chatStr, True, self.logging)

    def _calculate_cost(self, rates, dec=5):
        """Calculates approximate cost (USD) of LLM tokens generated to a given decimal place"""
        global total_cost
        if self.model in rates:
            prompt_rate, completion_rate = rates.get(self.model)
            prompt_cost = round((self.tokens["prompt"] * prompt_rate) / 1e6, dec)
            completion_cost = round(
                (self.tokens["completion"] * completion_rate) / 1e6, dec
            )
        else:
            prompt_cost = completion_cost = 0.0

        total_cost += round(prompt_cost + completion_cost, dec)
        self.cost["prompt"] += prompt_cost
        self.cost["completion"] += completion_cost

    def cost_report(self, dec=5):
        """Generates session cost report."""
        costStr = f"""Overall session cost: ${round(total_cost, dec)}

    Current agent using: {self.model}
        Subtotal: ${round(self.cost['prompt'] + self.cost['completion'], dec)}
        Input: ${self.cost['prompt']}
        Output: ${self.cost['completion']}
"""
        self._log_and_print(costStr, True, self.logging)

    def _condense_iterations(self, responses):
        """Condenses multiple API responses into a single coherent response."""
        responses = "\n\n".join(
            [
                "\n".join([f"Iteration: {i + 1}", responses[i]])
                for i in range(len(responses))
            ]
        )

        self._log_and_print(
            f"Using {self.small_model} to condense system responses...\n",
            self.verbose,
            self.logging,
        )
        condensed = self._init_chat_completion(
            model=self.small_model,
            prompt=text_library["modifiers"]["condense"] + "\n\n" + responses,
        )

        self._log_and_print(
            f"Condensed text:\n{condensed}\n", self.verbose, self.logging
        )

        return condensed

    def _refine_user_prompt(self, old_prompt):
        """Refines an LLM prompt using specified rewrite actions."""
        modifierDict = text_library["modifiers"]
        refineDict = text_library["refinement"]
        self._log_and_print(
            f"Using {self.small_model} to optimize initial user request...\n",
            True,
            self.logging,
        )
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
            model=self.small_model, prompt=updated_prompt
        )

        new_prompt = (
            self._condense_iterations(refined) if self.iterations > 1 else refined
        )

        self._log_and_print(
            f"Refined query prompt:\n{new_prompt}", self.verbose, self.logging
        )

        return new_prompt
