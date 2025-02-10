from propmptpal.core import CreateAgent
from openai import OpenAI

# Confirm environment API key
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    raise EnvironmentError("OPENAI_API_KEY environment variable not found!")

# Initialize OpenAI client and conversation chat
client = OpenAI(api_key=api_key)
chat = client.beta.threads.create()
chat.current_chat_calls = 0
client.chat_ids = set([chat.id])


# OpenAI sub-class
class ChatGPT(CreateAgent):
    """
    A handler for managing queries to the OpenAI API, including prompt preparation,
    API request submission, response processing, and logging.

    This class provides a flexible interface to interact with OpenAIs models, including
    text-based models (e.g., GPT-4) and image generation models (e.g., DALL-E). It supports
    features such as associative prompt refinement, chain-of-thought reasoning, code extraction,
    logging, unit testing, and much more.
    """
    def __init__(
        self,
        model="gemini-2.0-flash",
        threshold=0.55,
        **kwargs,  # Pass remaining arguments to the base class
    ):
        # Set default valid models for ChatGPT
        openai_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o1-mini",
            "o1-preview",
            "dall-e-3",
            "dall-e-2",
        ]
        # As of January 24, 2025
        openai_rates = {
            "gpt-4o": (2.5, 10),
            "gpt-4o-mini": (0.150, 0.600),
            "o1-mini": (3, 12),
            "o1-preview": (15, 60),
            "dall-e-3": (2.5, 0.040),
            "dall-e-2": (2.5, 0.040),
        }

        # Initialize the base class with all parameters
        super().__init__(model=model, valid_models=openai_models, **kwargs)
        self.small_model = "gpt-4o-mini"

        # Agent-specific chat params
        global chat
        self.chat_id = chat.id
        chat.message_limit = message_limit
        if self.new_chat == True:
            self.start_new_chat()

        # Update token counters
        global total_tokens
        if self.model not in total_tokens.keys():
            total_tokens[self.model] = {"prompt": 0, "completion": 0}

    def start_new_chat(self, context=None):
        """Start a new chat with only the current agent and adds previous context if needed."""
        global chat
        chat = client.beta.threads.create()
        chat.current_chat_calls = 0
        chat.message_limit = self.message_limit

        # Add previous context
        if context:
            previous_context = client.beta.threads.messages.create(
                chat_id=chat.id, role="user", content=context
            )

        global client
        client.chat_ids |= set([chat.id])
        self.chat_id = chat.id

        # Report
        self._log_and_print(
            f"New chat created and added to current agent: {self.chat_id}\n",
            self.verbose,
            self.logging,
        )

    def _init_chat_completion(self, prompt, model, role="user"):
        """Initialize and submit a single chat completion request"""
        message = [
            {"role": "user", "content": prompt},
            {"role": "system", "content": role},
        ]

        completion = client.chat.completions.create(
            model=model,
            messages=message,
            n=self.iterations,
            seed=self.seed,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        self._update_token_count(completion)
        self._calculate_cost(openai_rates)

        return completion

    def _update_token_count(self, response_obj):
        """Updates token count for prompt and completion."""
        global total_tokens
        total_tokens[self.model]["prompt"] += response_obj.usage.prompt_tokens
        total_tokens[self.model]["completion"] += response_obj.usage.completion_tokens
        # Agent-specific counts
        self.tokens["prompt"] += response_obj.usage.prompt_tokens
        self.tokens["completion"] += response_obj.usage.completion_tokens

    def chat_report(self):
        """Report active chats from current session"""
        chatStr = f"""Current session chats:
    {'\n\t'.join(client.chat_ids)}
"""
        self._log_and_print(chatStr, True, self.logging)

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
                tools=[{"type": "code_interpreter"}] if interpreter == True else [],
            )
            self.agent = agent.id
        except Exception as e:
            raise RuntimeError(f"Failed to create assistant: {e}")

    def _get_current_messages(self):
        """Fetches all messages from a chat in order and returns them as a text block."""
        messages = client.beta.threads.messages.list(chat_id=self.chat_id)
        sorted_messages = sorted(messages.data, key=lambda msg: msg.created_at)
        conversation = [x.content[0].text.value.strip() for x in sorted_messages]

        return "\n\n".join(conversation)

    def _send_chat_message(self) -> str:
        """
        Sends a user prompt to an existing chat, runs the assistant,
        and retrieves the response if successful.

        Returns:
            str: The text response from the assistant.

        Raises:
            ValueError: If the assistant fails to generate a response.
        """
        # Adds user prompt to existing chat.
        try:
            new_message = client.beta.threads.messages.create(
                chat_id=self.chat_id, role="user", content=self.prompt
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create message: {e}")

        # Run the assistant on the chat
        current_run = client.beta.threads.runs.create(
            chat_id=self.chat_id, assistant_id=self.agent
        )

        # Wait for completion and retrieve responses
        while True:
            self.run_status = client.beta.threads.runs.retrieve(
                chat_id=self.chat_id, run_id=current_run.id
            )
            if self.run_status.status in ["completed", "failed"]:
                break
            else:
                time.sleep(1)  # Wait before polling again

        if self.run_status.status == "completed":
            messages = client.beta.threads.messages.list(chat_id=self.chat_id)
            if messages.data:  # Check if messages list is not empty
                return messages.data[0].content[0].text.value
            else:
                raise ValueError("No messages found in the chat.")
        else:
            raise ValueError("Assistant failed to generate a response.")

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
        image_file = _check_unique_filename(image_file)
        with open(image_file, "wb") as outFile:
            outFile.write(image_data)

        self.last_message = (
            "\nRevised image prompt:\n"
            + response.data[0].revised_prompt
            + "\nGenerated image saved to:\n"
            + image_file
        )
        self._log_and_print(self.last_message, True, self.logging)

    def _validate_image_params(self, dimensions, quality):
        """Validates image dimensions and quality for the model."""
        valid_dimensions = {
            "dall-e-3": ["1024x1024", "1792x1024", "1024x1792"],
            "dall-e-2": ["1024x1024", "512x512", "256x256"],
        }
        if (
            self.model in valid_dimensions
            and dimensions.lower() not in valid_dimensions[self.model]
        ):
            self.dimensions = "1024x1024"
        else:
            self.dimensions = dimensions

        self.quality = (
            "hd"
            if quality.lower() in {"h", "hd", "high", "higher", "highest"}
            else "standard"
        )
        self.quality = (
            "hd" if self.label == "photographer" else self.quality
        )  # Check for photo role
