
import os
import re
import sys
import time
import string
import requests
from datetime import datetime
from collections import defaultdict

from openai import OpenAI

from promptpal.lib import text_library

roleDict = text_library["roles"]
modifierDict = text_library["modifiers"]
refineDict = text_library["refinement"]
extDict = text_library["extensions"]
patternDict = text_library["patterns"]

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
        message_limit (int): Maximum number of messages to a single thread before summarizing content and passing to new instance
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

    def __init__(
        self,
        logging = True,
        verbose = True,
        silent = False,
        refine = False,
        glyph = False,
        chain_of_thought = False,
        save_code = False,
        scan_dirs = False,
        new_thread = False,
        model = "gpt-4o-mini",
        role = "assistant",
        seed = "t634e``R75T86979UYIUHGVCXZ",
        iterations = 1,
        temperature = 0.7,
        top_p = 1.0,
        dimensions = "NA",
        quality = "NA",
        stage = 'normal',
        message_limit = 20):
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
        self.new_thread = new_thread
        self.model = model
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

        # Agent-specific thread params
        global thread
        self.thread_id = thread.id
        thread.message_limit = message_limit
        if self.new_thread == True:
            self.start_new_thread()

        # Update token counters
        global total_tokens
        self.cost = {"prompt": 0.0, "completion": 0.0}
        self.tokens = {"prompt": 0, "completion": 0}
        if self.model not in total_tokens.keys():
            total_tokens[self.model] = {"prompt": 0, "completion": 0}
        
        # Validdate specific hyperparams
        self.stage = self.stage if self.stage == 'refine_only' else 'normal'
        self.seed = self.seed if isinstance(self.seed, int) else self._string_to_binary(self.seed)
        self.temperature, self.top_p = self._validate_probability_params(self.temperature, self.top_p)
        
        # Validate user inputs
        self._prepare_system_role(role)
        self._validate_model_selection(model)
        if self.model in ["dall-e-2", "dall-e-3"]:
            self._validate_image_params(dimensions, quality)
        self._create_new_agent(interpreter=self.save_code)

        # Initialize reporting and related vars
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.prefix = f"{self.label}.{self.model.replace('-', '_')}.{self.timestamp}"        
        if self.logging: self._setup_logging()
        self._log_and_print(self.status(), False, self.logging)

    def _validate_types(self):
        """
        Validates the types of the instance attributes for CreateAgent.

        Raises:
            TypeError: If any attribute has an incorrect type.
            ValueError: If any integer attribute is not positive.
        """
        expected_types = {
            'logging': bool,
            'verbose': bool,
            'silent': bool,
            'refine_prompt': bool,
            'glyph_prompt': bool,
            'chain_of_thought': bool,
            'save_code': bool,
            'scan_dirs': bool,
            'new_thread': bool,
            'model': str,
            'role': str,
            'seed': (int, str),  # seed can be either int or str
            'iterations': int,
            'temperature': float,
            'top_p': float,
            'dimensions': str,
            'quality': str,
            'stage': str,
            'message_limit': int
        }

        for attr_name, expected_type in expected_types.items():
            value = getattr(self, attr_name, None)  # Get the attribute value from self
            if isinstance(expected_type, tuple):
                # Check if value matches any expected type in the tuple
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected type for {attr_name} is {expected_type}, got {type(value).__name__}")
            else:
                # Check if value matches the expected type
                if not isinstance(value, expected_type):
                    raise TypeError(f"Expected type for {attr_name} is {expected_type}, got {type(value).__name__}")
            
            # Check if integer-type values are positive
            if expected_type == int and value <= 0:
                raise ValueError(f"{attr_name} must be a positive integer, got {value}")

    def _setup_logging(self):
        """
        Prepare logging setup.
        """
        self.log_text = []
        self.log_file = f"logs/{self.prefix}.transcript.log"
        self.log_file = self._check_unique_filename(self.log_file)
        os.makedirs("logs", exist_ok=True)
        with open(self.log_file, "w") as f:
            f.write("New session initiated.\n")

    def _validate_probability_params(self, temp, topp):
        """Ensure temperature and top_p are valid"""
        # Acceptable ranges
        if temp < 0.0 or temp > 2.0:
            temp = 0.7
        if topp < 0.0 or topp > 2.0:
            topp = 1.0

        # Only one variable is changed at a time
        if temp != 0.7 and topp != 1.0:
            topp = 1.0

        return temp, topp

    def _prepare_query_text(self, prompt_text):
        """
        Prepares the query, including prompt modifications and image handling.
        """
        self.prompt = prompt_text

        # Identifies files to be read in
        files = self._find_existing_files()
        for f in files:
            self.prompt += "\n\n" + self._read_file_contents(f)
        if self.scan_dirs == True:
            paths = self._find_existing_paths()
            for d in paths:
                self.prompt += "\n\n" + self._scan_directory(d)

        # Refine prompt if required
        if self.refine_prompt or self.glyph_prompt:
            self._log_and_print(
                "\nAgent using gpt-4o-mini to optimize initial user request...\n", True, self.logging)
            self.prompt = self._refine_user_prompt(self.prompt)

    def _validate_model_selection(self, input_model):
        """Validates and selects the model based on user input or defaults."""
        openai_models = ["gpt-4o","o1","o1-mini","o1-preview","dall-e-3","dall-e-2"]
        self.model = input_model.lower() if input_model.lower() in openai_models else "gpt-4o-mini"

    def _prepare_system_role(self, input_role):
        """Prepares system role text."""

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
            self.role += modifierDict["cot"]

    def _read_file_contents(self, filename):
        """Reads the contents of a given file."""
        with open(filename, "r", encoding="utf-8") as f:
            return f"# File: {filename}\n{f.read()}"

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

    def _validate_image_params(self, dimensions, quality):
        """Validates image dimensions and quality for the model."""
        valid_dimensions = {"dall-e-3": ["1024x1024", "1792x1024", "1024x1792"],
                            "dall-e-2": ["1024x1024", "512x512", "256x256"]}
        if (self.model in valid_dimensions and dimensions.lower() not in valid_dimensions[self.model]):
            self.dimensions = "1024x1024"
        else:
            self.dimensions = dimensions

        self.quality = "hd" if quality.lower() in {"h", "hd", "high", "higher", "highest"} else "standard"
        self.quality = "hd" if self.label == "photographer" else self.quality # Check for photo role
        
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
    Assistant ID: {self.agent}
    Thread ID: {thread.id}
    Requests in current thread: {thread.current_thread_calls}
    """
        self._log_and_print(statusStr, True, self.logging)

        # Token usage report
        self.token_report()
        
        # $$$ report
        self.cost_report()

        # Thread report
        self.thread_report()

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

    def request(self, prompt=''):
        """Submits the query to OpenAIs API and processes the response."""
        # Checks for last system response is not prompt provided
        if prompt == '':
            try:
                prompt = self.last_message
            except Exception as e:
                raise ValueError(f"No existing messages found in thread: {e}")

        # Update user prompt 
        self._prepare_query_text(prompt)
        self._log_and_print(
            f"\n{self.role_name} using {self.model} to process updated conversation thread...\n",
                True, self.logging)

        if self.stage != "refine_only":
            if "dall-e" not in self.model:
                thread.current_thread_calls += 1
                self._handle_text_request()
            else:
                self._handle_image_request()

        # Check current scope thread
        if thread.current_thread_calls >= thread.message_limit:
            self._log_and_print(f"\nReached end of current thread limit.\n", self.verbose, False)
            summary = self.summarize_current_thread()
            self.start_new_thread("The following is a summary of a ongoing conversation with a user and an AI assistant:\n" + summary)

    def _init_chat_completion(self, prompt, model='gpt-4o-mini', role='user', iters=1, seed=42, temp=0.7, top_p=1.0):
        """Initialize and submit a single chat completion request"""
        message = [{"role": "user", "content": prompt}, {"role": "system", "content": role}]

        completion = client.chat.completions.create(
            model=model, messages=message, n=iters,
            seed=seed, temperature=temp, top_p=top_p)
        self._update_token_count(completion)
        self._calculate_cost()

        return completion

    def summarize_current_thread(self):
        """Summarize current conversation history for future context parsing."""
        self._log_and_print(f"\nAgent using gpt-4o-mini to summarize current thread...\n", self.verbose, False)

        # Get all thread messages
        all_messages = self._get_thread_messages()

        # Generate concise summary
        summary_prompt = modifierDict['summarize'] + "\n\n" + all_messages
        summarized = self._init_chat_completion(prompt=summary_prompt, iters=self.iterations, seed=self.seed)

        return summarized.choices[0].message.content.strip()

    def _get_thread_messages(self):
        """Fetches all messages from a thread in order and returns them as a text block."""
        messages = client.beta.threads.messages.list(thread_id=self.thread_id)
        sorted_messages = sorted(messages.data, key=lambda msg: msg.created_at)
        conversation = [x.content[0].text.value.strip() for x in sorted_messages]

        return "\n\n".join(conversation)

    @staticmethod
    def _check_unique_filename(filename):
        # Split the filename into name and extension
        name, ext = os.path.splitext(filename)
        counter = 1
        
        # Check if the file exists, and modify the name if it does
        while os.path.exists(filename):
            filename = f"{name}_{counter}{ext}"
            counter += 1
        
        return filename

    def _handle_text_request(self):
        """Processes text-based responses from OpenAIs chat models."""
        self.last_message = self._run_thread_request()
        self._update_token_count(self.run_status)
        self._calculate_cost()
        self._log_and_print(self.last_message, True, self.logging)

        # Extract code snippets
        code_snippets = self._extract_code_snippets()
        if self.save_code and len(code_snippets) > 0:
            self.code_files = []
            reportStr = "\nExtracted code saved to:\n"
            for lang in code_snippets.keys():
                code = code_snippets[lang]
                objects = self._extract_object_names(code, lang)
                file_name = f"{self._find_max_lines(code, objects)}.{self.timestamp}{extDict.get(lang, f'.{lang}')}".lstrip("_.")
                file_name = self._check_unique_filename(file_name)
                reportStr += f"\t{file_name}\n"
                self._write_script(code, file_name)

            self._log_and_print(reportStr, True, self.logging)

        # Check URL annotations - inactive for now
        #existing, not_existing = self._check_response_urls()
        #if len(not_existing) >= 1 or len(existing) >= 1:
        #    reportStr = "\nURL citations detecting in system message\n"
        #    if len(existing) >= 1:
        #        reportStr += 'Found:\n\t' '\n\t'.join(existing) + '\n'
        #    if len(not_existing) >= 1:
        #        reportStr += 'NOT found:\n\t' '\n\t'.join(not_existing) + '\n'
        #    self._log_and_print(reportStr, self.verbose, self.logging)

    def _write_script(self, content, file_name, outDir="code", lang=None):
        """Writes code to a file."""
        os.makedirs(outDir, exist_ok=True)
        self.code_files.append(f"{os.getcwd()}/{outDir}/{file_name}")
        with open(f"{outDir}/{file_name}", "w", encoding="utf-8") as f:
            if lang:
                f.write(f"#!/usr/bin/env {lang}\n\n")
            f.write(f"# Code generated by {self.model}\n\n")
            f.write(content)

    def _handle_image_request(self):
        """Processes image generation requests using OpenAIs image models."""
        os.makedirs("images", exist_ok=True)
        response = client.images.generate(
            model=self.model,
            prompt=self.prompt,
            n=1,
            size=self.dimensions,
            quality=self.quality,
        )
        self._update_token_count(response)
        self._calculate_cost()
        self._log_and_print(
            f"\nRevised initial prompt:\n{response.data[0].revised_prompt}",
            self.verbose,
            self.logging,
        )
        image_data = requests.get(response.data[0].url).content
        image_file = f"images/{self.prefix}.image.png"
        image_file = self._check_unique_filename(image_file)
        with open(image_file, "wb") as outFile:
            outFile.write(image_data)

        self.last_message = (
            "\nRevised image prompt:\n"
            + response.data[0].revised_prompt
            + "\nGenerated image saved to:\n"
            + image_file
        )
        self._log_and_print(self.last_message, True, self.logging)

    def _update_token_count(self, response_obj):
        """Updates token count for prompt and completion."""
        global total_tokens
        total_tokens[self.model]["prompt"] += response_obj.usage.prompt_tokens
        total_tokens[self.model]["completion"] += response_obj.usage.completion_tokens
        # Agent-specific counts
        self.tokens["prompt"] += response_obj.usage.prompt_tokens
        self.tokens["completion"] += response_obj.usage.completion_tokens

    def token_report(self):
        """Generates session token report."""
        allTokensStr = ""
        for x in total_tokens.keys():
            allTokensStr += f"{x}: Input = {total_tokens[x]['prompt']}; Completion = {total_tokens[x]['completion']}\n"

        tokenStr = f"""Overall session tokens:
    {allTokensStr}
    Current agent tokens: 
        Input: {self.tokens['prompt']}
        Output: {self.tokens['completion']}
"""
        self._log_and_print(tokenStr, True, self.logging)


    def thread_report(self):
        """Report active threads from current session"""
        threadStr = f"""Current session threads:
    {'\n\t'.join(client.thread_ids)}
"""
        self._log_and_print(threadStr, True, self.logging)

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

    def cost_report(self, dec=5):
        """Generates session cost report."""
        
        costStr = f"""Overall session cost: ${round(total_cost, dec)}

    Current agent using: {self.model}
        Subtotal: ${round(self.cost['prompt'] + self.cost['completion'], dec)}
        Input: ${self.cost['prompt']}
        Output: ${self.cost['completion']}
"""     
        self._log_and_print(costStr, True, self.logging)

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

    def _gen_iteration_str(self, responses):
        """Format single string with response iteration text"""
        outStr = "\n\n".join(
            [
                "\n".join([f"Iteration: {i + 1}", responses[i]])
                for i in range(len(responses))
            ]
        )
        self._log_and_print(outStr, self.verbose, self.logging)

        return outStr

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

    def _log_and_print(self, message, verb=True, log=True):
        """Logs and prints the provided message if verbose."""
        if message:
            if verb == True and self.silent == False:
                print(message)
            if log == True:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(message + "\n")

    @staticmethod
    def _string_to_binary(input_string):
        """Create a binary-like variable from a string for use a random seed"""
        # Convert all characters in a str to ASCII values and then to 8-bit binary
        binary = ''.join([format(ord(char), "08b") for char in input_string])
        # Constrain length
        return int(binary[0 : len(str(sys.maxsize))])

    @staticmethod
    def _is_code_file(file_path):
        """Check if a file has a code extension."""
        return os.path.splitext(file_path)[1].lower() in set(extDict.values())

    def _scan_directory(self, path="code"):
        """Recursively scan a directory and return the content of all code files."""
        codebase = ""
        for root, _, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                if self._is_code_file(file_path):
                    codebase += f"File: {file_path}\n"
                    codebase += self._read_file_contents(file_path)
                    codebase += "\n\n"

        return codebase

    def _find_existing_paths(self):
        """
        Scan the input string for existing paths and return them in separate lists.
        """
        # Regular expression to match potential file paths
        path_pattern = re.compile(r'([a-zA-Z]:\\[^:<>"|?\n]*|/[^:<>"|?\n]*)')

        # Find all matches in the input string
        matches = path_pattern.findall(self.prompt)

        # Separate files and directories
        existing_paths = []
        for match in matches:
            if os.path.isdir(match):
                existing_paths.append(match)

        return existing_paths

    def _find_existing_files(self):

        # Filter filenames by checking if they exist in the current directory or system's PATH
        existing_files = [
            x
            for x in self.prompt.split()
            if os.path.isfile(x.rstrip(string.punctuation))
        ]

        return existing_files

    def _extract_code_snippets(self):
        """
        Extract code snippets from a large body of text using triple backticks as delimiters.
        Also saves the language tag at the start of each snippet.
        """
        # Regular expression to match code blocks enclosed in triple backticks, including the language tag
        code_snippets = defaultdict(str)
        code_pattern = re.compile(r"```(\w+)\n(.*?)```", re.DOTALL)
        snippets = code_pattern.findall(self.last_message)
        for lang, code in snippets:
            code_snippets[lang] += code.strip()

        return code_snippets

    @staticmethod
    def _extract_object_names(code, language):
        """
        Extract defined object names (functions, classes, and variables) from a code snippet.
        """
        # Get language-specific patterns
        patterns = patternDict.get(language, {})

        # Extract object names using the language-specific patterns
        classes = patterns.get("class", re.compile(r"")).findall(code)
        functions = patterns.get("function", re.compile(r"")).findall(code)
        variables = patterns.get("variable", re.compile(r"")).findall(code)

        # Select objects to return based on hierarachy
        if len(classes) > 0:
            return classes
        elif len(functions) > 0:
            return functions
        else:
            return variables

    @staticmethod
    def _find_max_lines(code, object_names):
        """
        Count the number of lines of code for each object in the code snippet.

        Args:
            code (str): The code snippet to analyze.
            object_names (list): A list of object names to count lines for.

        Returns:
            str: Name of object with the largest line count.
        """
        rm_names = ["main", "functions", "classes", "variables"]
        line_counts = {name: 0 for name in object_names if name not in rm_names}
        line_counts['code'] = 1
        current_object = None

        for line in code.split("\n"):
            # Check if the line defines a new object
            for name in object_names:
                if re.match(rf"\s*(def|class)\s+{name}\s*[\(:]", line):
                    current_object = name
                    break

            # Count lines for the current object
            if current_object and line.strip() and current_object not in rm_names:
                line_counts[current_object] += 1

        return max(line_counts, key=line_counts.get)

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

    def _check_response_urls(self):
        """
        Extracts all URLs from the given text using regex,checks the existence of 
        each URL, and returns lists of existing and non-existing URLs.
        
        Args:
            text (str): The input text containing potential URLs.

        Returns:
            Tuple[List[str], List[str]]: A tuple containing two lists - 
                                           the first list for existing URLs,
                                           and the second for non-existing URLs.
        """
        # Define a regex pattern for URL extraction.
        url_pattern = r'https?://[^\s]+|ftp://[^\s]+'
        urls = re.findall(url_pattern, self.last_message)
        
        # Check if identified URLs are real
        existing_urls = non_existing_urls = []
        for url in urls:
            try:
                # Execute a curl command to check URL existence
                response = subprocess.run(['curl', '-Is', url], capture_output=True, text=True)
                
                if response.returncode == 0:
                    # Extract status code
                    status_line = response.stdout.splitlines()[0]
                    status_code = status_line.split()[1]
                    if status_code.startswith('2'):  # Status codes 2xx indicate success
                        existing_urls.append(url)
                    else:
                        non_existing_urls.append(url)
                else:
                    non_existing_urls.append(url)  # If curl fails, consider the URL as non-existing
            except Exception as e:
                non_existing_urls.append(url)  # Append URL in case of any exception

        return existing_urls, non_existing_urls
